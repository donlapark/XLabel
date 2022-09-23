# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

# Make ExplainableBoostingClassifier work on missing values.

from typing import DefaultDict

from interpret.privacy import (DPExplainableBoostingClassifier,
                               DPExplainableBoostingRegressor)
from interpret.utils import gen_perf_dicts
from interpret.glassbox.ebm.ebm import (BaseCoreEBM,
                                        EBMExplanation,
                                        EBMPreprocessor)
from interpret.glassbox.ebm.utils import DPUtils, EBMUtils
from interpret.glassbox.ebm.internal import Native
from interpret.glassbox.ebm.postprocessing import multiclass_postprocess
from interpret.utils import unify_data, unify_vector
from interpret.api.base import ExplainerMixin
from interpret.provider.compute import JobLibProvider
from interpret.utils import gen_name_from_class, gen_global_selector, gen_local_selector
import ctypes as ct

import numpy as np
from warnings import warn

from sklearn.base import is_classifier
from sklearn.utils.validation import check_is_fitted
import heapq

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin
)


class BaseEBM(BaseEstimator):
    """Client facing SK EBM."""

    # Interface modeled after:
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
    # https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html

    def __init__(
        self,
        # Explainer
        #
        # feature_names in scikit-learn convention should probably be passed in via the fit function.  Also,
        #   we can get feature_names via pandas dataframes, and those would only be known at fit time, so
        #   we need a version of feature_names_out_ with the underscore to indicate items set at fit time.
        #   Despite this, we need to recieve a list of feature_names here to be compatible with blackbox explainations
        #   where we still need to have feature_names, but we do not have a fit function since we explain existing
        #   models without fitting them ourselves.  To conform to a common explaination API we get the feature_names
        #   here.
        feature_names,
        # other packages LightGBM, CatBoost, Scikit-Learn (future) are using categorical specific ways to indicate
        #   feature_types.  The benefit to them is that they can accept multiple ways of specifying categoricals like:
        #   categorical = [true, false, true, true] OR categorical = [1, 4, 8] OR categorical = 'all'/'auto'/'none'
        #   We're choosing a different route because for visualization we want to be able to express multiple
        #   different types of data.  For example, if the user has data with strings of "low", "medium", "high"
        #   We want to keep both the ordinal nature of this feature and we wish to preserve the text for visualization
        #   scikit-learn callers can pre-convert these things to [0, 1, 2] in the correct order because they don't
        #   need to worry about visualizing the data afterwards, but for us we  need a way to specify the strings
        #   back anyways.  So we need some way to express both the categorical nature of features and the order
        #   mapping.  We can do this and more complicated conversions via:
        #   feature_types = ["categorical", ["low", "medium", "high"], "continuous", "time", "bool"]
        feature_types,
        # Data
        #
        # Ensemble
        outer_bags,
        inner_bags,
        # Core
        # TODO PK v.3 replace mains in favor of a "boosting stage plan"
        mains,
        interactions,
        validation_size,
        max_rounds,
        early_stopping_tolerance,
        early_stopping_rounds,
        # Native
        learning_rate,
        # Holte, R. C. (1993) "Very simple classification rules perform well on most commonly used datasets"
        # says use 6 as the minimum samples https://link.springer.com/content/pdf/10.1023/A:1022631118932.pdf
        # TODO PK try setting this (not here, but in our caller) to 6 and run tests to verify the best value.
        min_samples_leaf,
        max_leaves,
        # Overall
        n_jobs,
        random_state,
        # Preprocessor
        binning,
        max_bins,
        max_interaction_bins,
        # Differential Privacy
        epsilon=None,
        delta=None,
        composition=None,
        bin_budget_frac=None,
        privacy_schema=None,
    ):
        # NOTE: Per scikit-learn convention, we shouldn't attempt to sanity check these inputs here.  We just
        #       Store these values for future use.  Validate inputs in the fit or other functions.  More details in:
        #       https://scikit-learn.org/stable/developers/develop.html

        # Arguments for explainer
        self.feature_names = feature_names
        self.feature_types = feature_types

        # Arguments for ensemble
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags

        # Arguments for EBM beyond training a feature-step.
        self.mains = mains
        self.interactions = interactions
        self.validation_size = validation_size
        self.max_rounds = max_rounds
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_rounds = early_stopping_rounds

        # Arguments for internal EBM.
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaves = max_leaves

        # Arguments for overall
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Arguments for preprocessor
        self.binning = binning
        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins

        # Arguments for differential privacy
        self.epsilon = epsilon
        self.delta = delta
        self.composition = composition
        self.bin_budget_frac = bin_budget_frac
        self.privacy_schema = privacy_schema

    def fit(self, X, y, sample_weight=None):  # noqa: C901
        """ Fits model to provided samples.

        Args:
            X: Numpy array for training samples.
            y: Numpy array as training labels.
            sample_weight: Optional array of weights per sample. Should be same length as X and y.

        Returns:
            Itself.
        """

        # NOTE: Generally, we want to keep parameters in the __init__ function, since scikit-learn
        #       doesn't like parameters in the fit function, other than ones like weights that have
        #       the same length as the number of samples.  See:
        #       https://scikit-learn.org/stable/developers/develop.html
        #       https://github.com/microsoft/LightGBM/issues/2628#issue-536116395
        #


        # TODO PK sanity check all our inputs from the __init__ function, and this fit fuction

        # TODO PK we shouldn't expose our internal state until we are 100% sure that we succeeded
        #         so move everything to local variables until the end when we assign them to self.*

        # TODO PK we should do some basic checks here that X and y have the same dimensions and that
        #      they are well formed (look for NaNs, etc)

        # TODO PK handle calls where X.dim == 1.  This could occur if there was only 1 feature, or if
        #     there was only 1 sample?  We can differentiate either condition via y.dim and reshape
        #     AND add some tests for the X.dim == 1 scenario

        # TODO PK write an efficient striping converter for X that replaces unify_data for EBMs
        # algorithm: grap N columns and convert them to rows then process those by sending them to C

        # TODO: PK don't overwrite self.feature_names here (scikit-learn rules), and it's also confusing to
        #       user to have their fields overwritten.  Use feature_names_out_ or something similar
        X, y, self.feature_names, _ = unify_data(
            X, y, self.feature_names, self.feature_types, missing_data_allowed=True
        )

        # NOTE: Temporary override -- replace before push
        w = sample_weight if sample_weight is not None else np.ones_like(y, dtype=np.float64)
        w = unify_vector(w).astype(np.float64, casting="unsafe", copy=False)

        # Privacy calculations
        if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
            DPUtils.validate_eps_delta(self.epsilon, self.delta)
            DPUtils.validate_DP_EBM(self)

            if self.privacy_schema is None:
                warn("Possible privacy violation: assuming min/max values per feature/target are public info."
                     "Pass a privacy schema with known public ranges to avoid this warning.")
                self.privacy_schema = DPUtils.build_privacy_schema(X, y)

            self.domain_size_ = self.privacy_schema['target'][1] - self.privacy_schema['target'][0]

            # Split epsilon, delta budget for binning and learning
            bin_eps_ = self.epsilon * self.bin_budget_frac
            training_eps_ = self.epsilon - bin_eps_
            bin_delta_ = self.delta / 2
            training_delta_ = self.delta / 2
            
             # [DP] Calculate how much noise will be applied to each iteration of the algorithm
            if self.composition == 'classic':
                self.noise_scale_ = DPUtils.calc_classic_noise_multi(
                    total_queries = self.max_rounds * X.shape[1] * self.outer_bags, 
                    target_epsilon = training_eps_, 
                    delta = training_delta_, 
                    sensitivity = self.domain_size_ * self.learning_rate * np.max(w)
                )
            elif self.composition == 'gdp':
                self.noise_scale_ = DPUtils.calc_gdp_noise_multi(
                    total_queries = self.max_rounds * X.shape[1] * self.outer_bags, 
                    target_epsilon = training_eps_, 
                    delta = training_delta_
                )
                self.noise_scale_ = self.noise_scale_ * self.domain_size_ * self.learning_rate * np.max(w) # Alg Line 17
            else:
                raise NotImplementedError(f"Unknown composition method provided: {self.composition}. Please use 'gdp' or 'classic'.")
        else:
            bin_eps_, bin_delta_ = None, None
            training_eps_, training_delta_ = None, None

        # Build preprocessor
        self.preprocessor_ = EBMPreprocessor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            max_bins=self.max_bins,
            binning=self.binning,
            epsilon=bin_eps_, # Only defined during private training
            delta=bin_delta_,
            privacy_schema=getattr(self, 'privacy_schema', None)
        )
        self.preprocessor_.fit(X)
        X_orig = X
        X = self.preprocessor_.transform(X_orig)

        features_categorical = np.array([x == "categorical" for x in self.preprocessor_.col_types_], dtype=ct.c_int64)
        features_bin_count = np.array([len(x) for x in self.preprocessor_.col_bin_counts_], dtype=ct.c_int64)

        # NOTE: [DP] Passthrough to lower level layers for noise addition
        bin_data_counts = {i : self.preprocessor_.col_bin_counts_[i] for i in range(X.shape[1])}

        if self.interactions != 0:
            self.pair_preprocessor_ = EBMPreprocessor(
                feature_names=self.feature_names,
                feature_types=self.feature_types,
                max_bins=self.max_interaction_bins,
                binning=self.binning,
            )
            self.pair_preprocessor_.fit(X_orig)
            X_pair = self.pair_preprocessor_.transform(X_orig)
            pair_features_categorical = np.array([x == "categorical" for x in self.pair_preprocessor_.col_types_], dtype=ct.c_int64)
            pair_features_bin_count = np.array([len(x) for x in self.pair_preprocessor_.col_bin_counts_], dtype=ct.c_int64)
        else:
            self.pair_preprocessor_, X_pair, pair_features_categorical, pair_features_bin_count = None, None, None, None


        estimators = []
        seed = EBMUtils.normalize_initial_random_seed(self.random_state)

        native = Native.get_native_singleton()
        if is_classifier(self):
            self.classes_, y = np.unique(y, return_inverse=True)
            self._class_idx_ = {x: index for index, x in enumerate(self.classes_)}

            y = y.astype(np.int64, casting="unsafe", copy=False)
            n_classes = len(self.classes_)
            if n_classes > 2:  # pragma: no cover
                warn("Multiclass is still experimental. Subject to change per release.")
            if n_classes > 2 and self.interactions != 0:
                self.interactions = 0
                warn("Detected multiclass problem: forcing interactions to 0")
            for i in range(self.outer_bags):
                seed=native.generate_random_number(seed, 1416147523)
                estimator = BaseCoreEBM(
                    # Data
                    model_type="classification",
                    features_categorical=features_categorical,
                    features_bin_count=features_bin_count,
                    pair_features_categorical=pair_features_categorical,
                    pair_features_bin_count=pair_features_bin_count,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_samples_leaf,
                    max_leaves=self.max_leaves,
                    # Overall
                    random_state=seed,
                    # Differential Privacy
                    noise_scale=getattr(self, 'noise_scale_', None),
                    bin_counts=bin_data_counts,
                )
                estimators.append(estimator)
        else:
            n_classes = -1
            y = y.astype(np.float64, casting="unsafe", copy=False)
            for i in range(self.outer_bags):
                seed=native.generate_random_number(seed, 1416147523)
                estimator = BaseCoreEBM(
                    # Data
                    model_type="regression",
                    features_categorical=features_categorical,
                    features_bin_count=features_bin_count,
                    pair_features_categorical=pair_features_categorical,
                    pair_features_bin_count=pair_features_bin_count,
                    # Core
                    main_features=self.mains,
                    interactions=self.interactions,
                    validation_size=self.validation_size,
                    max_rounds=self.max_rounds,
                    early_stopping_tolerance=self.early_stopping_tolerance,
                    early_stopping_rounds=self.early_stopping_rounds,
                    # Native
                    inner_bags=self.inner_bags,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_samples_leaf,
                    max_leaves=self.max_leaves,
                    # Overall
                    random_state=seed,
                    # Differential Privacy
                    noise_scale=getattr(self, 'noise_scale_', None),
                    bin_counts=bin_data_counts,
                )
                estimators.append(estimator)

        # Train base models for main effects, pair detection.

        # scikit-learn returns an np.array for classification and
        # a single float64 for regression, so we do the same
        if is_classifier(self):
            self.intercept_ = np.zeros(
                Native.get_count_scores_c(n_classes), dtype=np.float64, order="C",
            )
        else:
            self.intercept_ = np.float64(0)

        provider = JobLibProvider(n_jobs=self.n_jobs)

        train_model_args_iter = (
            (estimators[i], X, y, w, X_pair, n_classes) for i in range(self.outer_bags)
        )

        estimators = provider.parallel(BaseCoreEBM.fit_parallel, train_model_args_iter)

        def select_pairs_from_fast(estimators, n_interactions):
            # Average rank from estimators
            pair_ranks = {}

            for n, estimator in enumerate(estimators):
                for rank, indices in enumerate(estimator.inter_indices_):
                    old_mean = pair_ranks.get(indices, 0)
                    pair_ranks[indices] = old_mean + ((rank - old_mean) / (n + 1))

            final_ranks = []
            total_interactions = 0
            for indices in pair_ranks:
                heapq.heappush(final_ranks, (pair_ranks[indices], indices))
                total_interactions += 1

            n_interactions = min(n_interactions, total_interactions)
            top_pairs = [heapq.heappop(final_ranks)[1] for _ in range(n_interactions)]
            return top_pairs

        if isinstance(self.interactions, int) and self.interactions > 0:
            # Select merged pairs
            pair_indices = select_pairs_from_fast(estimators, self.interactions)

            for estimator in estimators:
                # Discard initial interactions
                new_model = []
                new_feature_groups = []
                for i, feature_group in enumerate(estimator.feature_groups_):
                    if len(feature_group) != 1:
                        continue
                    new_model.append(estimator.model_[i])
                    new_feature_groups.append(estimator.feature_groups_[i])
                estimator.model_ = new_model
                estimator.feature_groups_ = new_feature_groups
                estimator.inter_episode_idx_ = 0

            if len(pair_indices) != 0:
                # Retrain interactions for base models

                staged_fit_args_iter = (
                    (estimators[i], X, y, w, X_pair, pair_indices) for i in range(self.outer_bags)
                )

                estimators = provider.parallel(BaseCoreEBM.staged_fit_interactions_parallel, staged_fit_args_iter)
        elif isinstance(self.interactions, int) and self.interactions == 0:
            pair_indices = []
        elif isinstance(self.interactions, list):
            pair_indices = self.interactions
            if len(pair_indices) != 0:
                # Check and remove duplicate interaction terms
                existing_terms = set()
                unique_terms = []

                for i, term in enumerate(pair_indices):
                    sorted_tuple = tuple(sorted(term))
                    if sorted_tuple not in existing_terms:
                        existing_terms.add(sorted_tuple)
                        unique_terms.append(term)

                # Warn the users that we have made change to the interactions list
                if len(unique_terms) != len(pair_indices):
                    warn("Detected duplicate interaction terms: removing duplicate interaction terms")
                    pair_indices = unique_terms
                    self.interactions = pair_indices

                # Retrain interactions for base models
                staged_fit_args_iter = (
                    (estimators[i], X, y, w, X_pair, pair_indices) for i in range(self.outer_bags)
                )

                estimators = provider.parallel(BaseCoreEBM.staged_fit_interactions_parallel, staged_fit_args_iter)
        else:  # pragma: no cover
            raise RuntimeError("Argument 'interaction' has invalid value")

        X = np.ascontiguousarray(X.T)
        if X_pair is not None:
            X_pair = np.ascontiguousarray(X_pair.T) # I have no idea if we're supposed to do this.

        if isinstance(self.mains, str) and self.mains == "all":
            main_indices = [[x] for x in range(X.shape[0])]
        elif isinstance(self.mains, list) and all(
            isinstance(x, int) for x in self.mains
        ):
            main_indices = [[x] for x in self.mains]
        else:  # pragma: no cover
            msg = "Argument 'mains' has invalid value (valid values are 'all'|list<int>): {}".format(
                self.mains
            )
            raise RuntimeError(msg)

        self.feature_groups_ = main_indices + pair_indices

        self.bagged_models_ = estimators
        # Merge estimators into one.
        self.additive_terms_ = []
        self.term_standard_deviations_ = []
        for index, _ in enumerate(self.feature_groups_):
            log_odds_tensors = []
            for estimator in estimators:
                log_odds_tensors.append(estimator.model_[index])

            averaged_model = np.average(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            self.additive_terms_.append(averaged_model)
            self.term_standard_deviations_.append(model_errors)

        # Get episode indexes for base estimators.
        main_episode_idxs = []
        inter_episode_idxs = []
        for estimator in estimators:
            main_episode_idxs.append(estimator.main_episode_idx_)
            inter_episode_idxs.append(estimator.inter_episode_idx_)

        self.breakpoint_iteration_ = [main_episode_idxs]
        if len(pair_indices) != 0:
            self.breakpoint_iteration_.append(inter_episode_idxs)

        # Extract feature group names and feature group types.
        # TODO PK v.3 don't overwrite feature_names and feature_types.  Create new fields called feature_names_out and
        #             feature_types_out_ or feature_group_names_ and feature_group_types_
        self.feature_names = []
        self.feature_types = []
        for index, feature_indices in enumerate(self.feature_groups_):
            feature_group_name = EBMUtils.gen_feature_group_name(
                feature_indices, self.preprocessor_.col_names_
            )
            feature_group_type = EBMUtils.gen_feature_group_type(
                feature_indices, self.preprocessor_.col_types_
            )
            self.feature_types.append(feature_group_type)
            self.feature_names.append(feature_group_name)

        if n_classes <= 2:
            if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
                # DP method of centering graphs can generalize if we log pairwise densities
                # No additional privacy loss from this step
                # self.additive_terms_ and self.preprocessor_.col_bin_counts_ are noisy and published publicly
                self._original_term_means_ = []
                for set_idx in range(len(self.feature_groups_)):
                    score_mean = np.average(self.additive_terms_[set_idx], weights=self.preprocessor_.col_bin_counts_[set_idx])
                    self.additive_terms_[set_idx] = (
                        self.additive_terms_[set_idx] - score_mean
                    )

                    # Add mean center adjustment back to intercept
                    self.intercept_ += score_mean
                    self._original_term_means_.append(score_mean)
            else:       
                # Mean center graphs - only for binary classification and regression
                scores_gen = EBMUtils.scores_by_feature_group(
                    X, X_pair, self.feature_groups_, self.additive_terms_
                )
                self._original_term_means_ = []

                for set_idx, _, scores in scores_gen:
                    score_mean = np.average(scores, weights=w)

                    self.additive_terms_[set_idx] = (
                        self.additive_terms_[set_idx] - score_mean
                    )

                    # Add mean center adjustment back to intercept
                    self.intercept_ += score_mean
                    self._original_term_means_.append(score_mean)
        else:
            # Postprocess model graphs for multiclass

            # Currently pairwise interactions are unsupported for multiclass-classification.
            binned_predict_proba = lambda x: EBMUtils.classifier_predict_proba(
                x, None, self.feature_groups_, self.additive_terms_, self.intercept_
            )

            postprocessed = multiclass_postprocess(
                X, self.additive_terms_, binned_predict_proba, self.feature_types
            )
            self.additive_terms_ = postprocessed["feature_graphs"]
            self.intercept_ = postprocessed["intercepts"]

        for feature_group_idx, feature_group in enumerate(self.feature_groups_):
            entire_tensor = [slice(None, None, None) for i in range(self.additive_terms_[feature_group_idx].ndim)]
            for dimension_idx, feature_idx in enumerate(feature_group):
                if self.preprocessor_.col_bin_counts_[feature_idx][0] == 0:
                    zero_dimension = entire_tensor.copy()
                    zero_dimension[dimension_idx] = 0
                    self.additive_terms_[feature_group_idx][tuple(zero_dimension)] = 0
                    self.term_standard_deviations_[feature_group_idx][tuple(zero_dimension)] = 0

        # Generate overall importance
        self.feature_importances_ = []
        if isinstance(self, (DPExplainableBoostingClassifier, DPExplainableBoostingRegressor)):
            # DP method of generating feature importances can generalize to non-dp if preprocessors start tracking joint distributions
            for i in range(len(self.feature_groups_)):
                mean_abs_score = np.average(np.abs(self.additive_terms_[i]), weights=self.preprocessor_.col_bin_counts_[i])
                self.feature_importances_.append(mean_abs_score)
        else:
            scores_gen = EBMUtils.scores_by_feature_group(
                X, X_pair, self.feature_groups_, self.additive_terms_
            )
            for set_idx, _, scores in scores_gen:
                mean_abs_score = np.mean(np.abs(scores))
                self.feature_importances_.append(mean_abs_score)

        # Generate selector
        # TODO PK v.3 shouldn't this be self._global_selector_ ??
        self.global_selector = gen_global_selector(
            X_orig, self.feature_names, self.feature_types, None
        )

        self.has_fitted_ = True
        return self

    # Select pairs from base models
    def _merged_pair_score_fn(self, model_type, X, y, X_pair, feature_groups, model, intercept):
        if model_type == "classification":
            prob = EBMUtils.classifier_predict_proba(
                X, X_pair, feature_groups, model, intercept
            )
            return (
                0 if len(y) == 0 else log_loss(y, prob)
            )  # use logloss to conform consistnetly and for multiclass
        elif model_type == "regression":
            pred = EBMUtils.regressor_predict(
                X, X_pair, feature_groups, model, intercept
            )
            return 0 if len(y) == 0 else mean_squared_error(y, pred)
        else:  # pragma: no cover
            msg = "Unknown model_type: '{}'.".format(model_type)
            raise ValueError(msg)

    def decision_function(self, X):
        """ Predict scores from model before calling the link function.

            Args:
                X: Numpy array for samples.

            Returns:
                The sum of the additive term contributions.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        decision_scores = EBMUtils.decision_function(
            X, X_pair, self.feature_groups_, self.additive_terms_, self.intercept_
        )

        return decision_scores

    def explain_global(self, name=None):
        """ Provides global explanation for model.

        Args:
            name: User-defined explanation name.

        Returns:
            An explanation object,
            visualizing feature-value pairs as horizontal bar chart.
        """
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        # Obtain min/max for model scores
        lower_bound = np.inf
        upper_bound = -np.inf
        for feature_group_index, _ in enumerate(self.feature_groups_):
            errors = self.term_standard_deviations_[feature_group_index]
            scores = self.additive_terms_[feature_group_index]

            lower_bound = min(lower_bound, np.min(scores - errors))
            upper_bound = max(upper_bound, np.max(scores + errors))

        bounds = (lower_bound, upper_bound)

        # Add per feature graph
        data_dicts = []
        feature_list = []
        density_list = []
        for feature_group_index, feature_indexes in enumerate(
            self.feature_groups_
        ):
            model_graph = self.additive_terms_[feature_group_index]

            # NOTE: This uses stddev. for bounds, consider issue warnings.
            errors = self.term_standard_deviations_[feature_group_index]

            if len(feature_indexes) == 1:
                # hack. remove the 0th index which is for missing values
                model_graph = model_graph[1:]
                errors = errors[1:]


                bin_labels = self.preprocessor_._get_bin_labels(feature_indexes[0])
                # bin_counts = self.preprocessor_.get_bin_counts(
                #     feature_indexes[0]
                # )
                scores = list(model_graph)
                upper_bounds = list(model_graph + errors)
                lower_bounds = list(model_graph - errors)
                density_dict = {
                    "names": self.preprocessor_._get_hist_edges(feature_indexes[0]),
                    "scores": self.preprocessor_._get_hist_counts(feature_indexes[0]),
                }

                feature_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": scores,
                    "scores_range": bounds,
                    "upper_bounds": upper_bounds,
                    "lower_bounds": lower_bounds,
                }
                feature_list.append(feature_dict)
                density_list.append(density_dict)

                data_dict = {
                    "type": "univariate",
                    "names": bin_labels,
                    "scores": model_graph,
                    "scores_range": bounds,
                    "upper_bounds": model_graph + errors,
                    "lower_bounds": model_graph - errors,
                    "density": {
                        "names": self.preprocessor_._get_hist_edges(feature_indexes[0]),
                        "scores": self.preprocessor_._get_hist_counts(
                            feature_indexes[0]
                        ),
                    },
                }
                if is_classifier(self):
                    data_dict["meta"] = {
                        "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                    }

                data_dicts.append(data_dict)
            elif len(feature_indexes) == 2:
                # hack. remove the 0th index which is for missing values
                model_graph = model_graph[1:, 1:]
                # errors = errors[1:, 1:]  # NOTE: This is commented as it's not used in this branch.


                bin_labels_left = self.pair_preprocessor_._get_bin_labels(feature_indexes[0])
                bin_labels_right = self.pair_preprocessor_._get_bin_labels(feature_indexes[1])

                feature_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                feature_list.append(feature_dict)
                density_list.append({})

                data_dict = {
                    "type": "interaction",
                    "left_names": bin_labels_left,
                    "right_names": bin_labels_right,
                    "scores": model_graph,
                    "scores_range": bounds,
                }
                data_dicts.append(data_dict)
            else:  # pragma: no cover
                raise Exception("Interactions greater than 2 not supported.")

        overall_dict = {
            "type": "univariate",
            "names": self.feature_names,
            "scores": self.feature_importances_,
        }
        internal_obj = {
            "overall": overall_dict,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_global",
                    "value": {"feature_list": feature_list},
                },
                {"explanation_type": "density", "value": {"density": density_list}},
            ],
        }

        return EBMExplanation(
            "global",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=self.global_selector,
        )

    def explain_local(self, X, y=None, name=None):
        """ Provides local explanations for provided samples.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each sample as horizontal bar charts.
        """

        # Produce feature value pairs for each sample.
        # Values are the model graph score per respective feature group.
        if name is None:
            name = gen_name_from_class(self)

        check_is_fitted(self, "has_fitted_")

        X, y, _, _ = unify_data(X, y, self.feature_names, self.feature_types, missing_data_allowed=True)

        # Transform y if classifier
        if is_classifier(self) and y is not None:
            y = np.array([self._class_idx_[el] for el in y])

        samples = self.preprocessor_.transform(X)
        samples = np.ascontiguousarray(samples.T)

        if self.interactions != 0:
            pair_samples = self.pair_preprocessor_.transform(X)
            pair_samples = np.ascontiguousarray(pair_samples.T)
        else:
            pair_samples = None

        scores_gen = EBMUtils.scores_by_feature_group(
            samples, pair_samples, self.feature_groups_, self.additive_terms_
        )

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        n_rows = samples.shape[1]
        data_dicts = []
        intercept = self.intercept_
        if not is_classifier(self) or len(self.classes_) <= 2:
            if isinstance(self.intercept_, np.ndarray) or isinstance(
                self.intercept_, list
            ):
                intercept = intercept[0]

        for _ in range(n_rows):
            data_dict = {
                "type": "univariate",
                "names": [],
                "scores": [],
                "values": [],
                "extra": {"names": ["Intercept"], "scores": [intercept], "values": [1]},
            }
            if is_classifier(self):
                data_dict["meta"] = {
                    "label_names": self.classes_.tolist()  # Classes should be numpy array, convert to list.
                }
            data_dicts.append(data_dict)

        for set_idx, feature_group, scores in scores_gen:
            for row_idx in range(n_rows):
                feature_name = self.feature_names[set_idx]
                data_dicts[row_idx]["names"].append(feature_name)
                data_dicts[row_idx]["scores"].append(scores[row_idx])
                if len(feature_group) == 1:
                    data_dicts[row_idx]["values"].append(
                        X[row_idx, feature_group[0]]
                    )
                else:
                    data_dicts[row_idx]["values"].append("")

        is_classification = is_classifier(self)
        if is_classification:
            scores = EBMUtils.classifier_predict_proba(
                samples, pair_samples, self.feature_groups_, self.additive_terms_, self.intercept_,
            )
        else:
            scores = EBMUtils.regressor_predict(
                samples, pair_samples, self.feature_groups_, self.additive_terms_, self.intercept_,
            )

        perf_list = []
        perf_dicts = gen_perf_dicts(scores, y, is_classification)
        for row_idx in range(n_rows):
            perf = None if perf_dicts is None else perf_dicts[row_idx]
            perf_list.append(perf)
            data_dicts[row_idx]["perf"] = perf

        selector = gen_local_selector(data_dicts, is_classification=is_classification)


        additive_terms = []
        for feature_group_index, feature_indexes in enumerate(self.feature_groups_):
            if len(feature_indexes) == 1:
                # hack. remove the 0th index which is for missing values
                additive_terms.append(self.additive_terms_[feature_group_index][1:])
            elif len(feature_indexes) == 2:
                # hack. remove the 0th index which is for missing values
                additive_terms.append(self.additive_terms_[feature_group_index][1:, 1:])
            else:
                raise ValueError("only handles 1D/2D")

        internal_obj = {
            "overall": None,
            "specific": data_dicts,
            "mli": [
                {
                    "explanation_type": "ebm_local",
                    "value": {
                        "scores": additive_terms,
                        "intercept": self.intercept_,
                        "perf": perf_list,
                    },
                }
            ],
        }
        internal_obj["mli"].append(
            {
                "explanation_type": "evaluation_dataset",
                "value": {"dataset_x": X, "dataset_y": y},
            }
        )

        return EBMExplanation(
            "local",
            internal_obj,
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            name=name,
            selector=selector,
        )


class ExplainableBoostingClassifier(BaseEBM, ClassifierMixin, ExplainerMixin):
    """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog. """

    # TODO PK v.3 use underscores here like ClassifierMixin._estimator_type?
    available_explanations = ["global", "local"]
    explainer_type = "model"

    """ Public facing EBM classifier."""

    def __init__(
        self,
        # Explainer
        feature_names=None,
        feature_types=None,
        # Preprocessor
        max_bins=256,
        max_interaction_bins=32,
        binning="quantile",
        # Stages
        mains="all",
        interactions=10,
        # Ensemble
        outer_bags=8,
        inner_bags=0,
        # Boosting
        learning_rate=0.01,
        validation_size=0.15,
        early_stopping_rounds=50,
        early_stopping_tolerance=1e-4,
        max_rounds=5000,
        # Trees
        min_samples_leaf=2,
        max_leaves=3,
        # Overall
        n_jobs=-2,
        random_state=42,
    ):
        """ Explainable Boosting Classifier. The arguments will change in a future release, watch the changelog.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            max_bins: Max number of bins per feature for pre-processing stage.
            max_interaction_bins: Max number of bins per feature for pre-processing stage on interaction terms. Only used if interactions is non-zero.
            binning: Method to bin values for pre-processing. Choose "uniform", "quantile" or "quantile_humanized".
            mains: Features to be trained on in main effects stage. Either "all" or a list of feature indexes.
            interactions: Interactions to be trained on.
                Either a list of lists of feature indices, or an integer for number of automatically detected interactions.
                Interactions are forcefully set to 0 for multiclass problems.
            outer_bags: Number of outer bags.
            inner_bags: Number of inner bags.
            learning_rate: Learning rate for boosting.
            validation_size: Validation set size for boosting.
            early_stopping_rounds: Number of rounds of no improvement to trigger early stopping.
            early_stopping_tolerance: Tolerance that dictates the smallest delta required to be considered an improvement.
            max_rounds: Number of rounds for boosting.
            min_samples_leaf: Minimum number of cases for tree splits used in boosting.
            max_leaves: Maximum leaf nodes used in boosting.
            n_jobs: Number of jobs to run in parallel.
            random_state: Random state.
        """
        super(ExplainableBoostingClassifier, self).__init__(
            # Explainer
            feature_names=feature_names,
            feature_types=feature_types,
            # Preprocessor
            max_bins=max_bins,
            max_interaction_bins=max_interaction_bins,
            binning=binning,
            # Stages
            mains=mains,
            interactions=interactions,
            # Ensemble
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            # Boosting
            learning_rate=learning_rate,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_tolerance=early_stopping_tolerance,
            max_rounds=max_rounds,
            # Trees
            min_samples_leaf=min_samples_leaf,
            max_leaves=max_leaves,
            # Overall
            n_jobs=n_jobs,
            random_state=random_state,
        )

    # TODO: Throw ValueError like scikit for 1d instead of 2d arrays
    def predict_proba(self, X):
        """ Probability estimates on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Probability estimate of sample for each class.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        prob = EBMUtils.classifier_predict_proba(
            X, X_pair, self.feature_groups_, self.additive_terms_, self.intercept_
        )
        return prob

    def predict(self, X):
        """ Predicts on provided samples.

        Args:
            X: Numpy array for samples.

        Returns:
            Predicted class label per sample.
        """
        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(X, None, self.feature_names, self.feature_types, missing_data_allowed=True)
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        # TODO PK add a test to see if we handle X.ndim == 1 (or should we throw ValueError)

        return EBMUtils.classifier_predict(
            X,
            X_pair,
            self.feature_groups_,
            self.additive_terms_,
            self.intercept_,
            self.classes_,
        )

    def predict_and_contrib(self, X, output='probabilities'):
        """Predicts on provided samples, returning predictions and explanations for each sample.

        Args:
            X: Numpy array for samples.
            output: Prediction type to output (i.e. one of 'probabilities', 'logits', 'labels')

        Returns:
            Predictions and local explanations for each sample.
        """

        allowed_outputs = ['probabilities', 'logits', 'labels']
        if output not in allowed_outputs:
            msg = "Argument 'output' has invalid value.  Got '{}', expected one of " 
            + repr(allowed_outputs)
            raise ValueError(msg.format(output))

        check_is_fitted(self, "has_fitted_")
        X_orig, _, _, _ = unify_data(
            X, None, self.feature_names, self.feature_types, missing_data_allowed=True
        )
        X = self.preprocessor_.transform(X_orig)
        X = np.ascontiguousarray(X.T)

        if self.interactions != 0:
            X_pair = self.pair_preprocessor_.transform(X_orig)
            X_pair = np.ascontiguousarray(X_pair.T)
        else:
            X_pair = None

        return EBMUtils.classifier_predict_and_contrib(
            X,
            X_pair,
            self.feature_groups_,
            self.additive_terms_,
            self.intercept_,
            self.classes_,
            output)
