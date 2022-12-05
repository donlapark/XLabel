"""A Streamlit app designed to help with data labeling with explainable
machine learning approach. It can handle data with many labels and many
classes. For each label, an Explainable Boosting Machine model is
trained on the labeled data, then it makes class predictions and
provides per-instance local explanations, which are then used to make
heatmaps, displayed in the main screen.
"""
import json
import math
import os
import pickle as pkle

from interpret.glassbox.ebm.ebm import ExplainableBoostingClassifier

import numpy as np
import pandas as pd

import altair as alt
import streamlit as st
from streamlit import session_state as _state
import streamlit.components.v1 as components

_PERSIST_STATE_KEY = f"{__name__}_PERSIST"
_CONFIGS_FILE = "configs.json"
_MODEL = "_saved_models.pickle"
_NUM_FEAT_PER_ROW = 11

st.set_page_config(layout="wide")


def main():
    """The main Streamlit app."""
    if "configs" not in _state:
        try:
            with open(_CONFIGS_FILE, "r") as _file:
                _state["configs"] = json.load(_file)
        except FileNotFoundError:
            create_config_file()

        _state["loaded_new_file"] = True

    st.sidebar.write("Current database: " + _state.configs["db_filename"])

    st.sidebar.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="uploaded_files",
        accept_multiple_files=False,
        on_change=update_file
        )

    with st.sidebar.form("sidebar"):
        st.slider(
            "Number of labels",
            min_value=1,
            max_value=20,
            value=_state.configs["sidebar"]["num_labels"],
            step=1,
            key="num_labels"
        )

        st.selectbox(
            "Include data with label/prediction mismatches?",
            ("Yes", "No"),
            key="relabel",
            index=("Yes", "No").index(_state.configs["sidebar"]["relabel"])
        )

        st.selectbox(
            "Sampling mode",
            ("Fixed sample size", "Confidence threshold"),
            key="mode",
            index=("Fixed sample size",
                   "Confidence threshold").index(_state.configs["sidebar"]["mode"])
        )

        st.slider(
            "Sample size (for \"Fixed sample size\" mode)",
            min_value=1,
            max_value=500,
            value=_state.configs["sidebar"]["n_samples"],
            step=1,
            key="n_samples"
        )

        st.slider(
            "Threshold (for \"Confidence threshold\" mode)",
            min_value=0.00,
            max_value=1.00,
            value=_state.configs["sidebar"]["threshold"],
            step=0.01,
            format="%.2f",
            key="threshold"
        )

        form_cols = st.columns((2.2, 1, 4))
        form_cols[1].form_submit_button("Sample", on_click=sample_and_predict)

    if "pages" in _state:
        st.sidebar.radio('Labels',
                         _state.pages,
                         key="label_page",
                         index=_state.next_clicked,
                         on_change=update_counter
                         )

        label = _state.label_page
        display_main_screen(label)


def update_file():
    """Update the state parameters after a file has been uploaded"""
    if _state.uploaded_files is not None:
        _state.configs["db_filename"] = _state.uploaded_files.name
        _state.loaded_new_file = True


def init_state_params():
    """Initialize all state parameters.

    This function will be called by sample_and_predict() when
    _state.pages has not been initialized.

    State parameters:
        database: The pandas dataframe of the database.
        configs: The saved configs of sidebar widgets.
        pages: The list of all labels.
        classes: The dict of all classes for each label.
        class_to_num: The encoding dict of classes into integers.
        num_to_class: The decoding dict of integers into classes.
        class_to_num: The dict that maps each class to the
          corresponding number.
        previous_: The index of the previous page.
        next_: The index of the next page.
        next_clicked: The index of the current page.
        counter: A dummy used to go to the top of a new screen.
        local_results: A dict of outputs of EBM
          used to write predictions and plot heatmaps on screen.
        models: A dict of EBM models to predict the labels.
          the models will be loaded and saved in pickle format.
        predictions: A pandas dataframe; each column contains EBM's
          predictions of each label.
        unlabeled_index: A pandas index of unlabeled rows. When new
          labels are added to the database, compute_unlabeled_index()
          needs to be called to track the changes.
    """
    if _state.uploaded_files is not None:
        data_file = _state.uploaded_files
        _state.configs["db_filename"] = data_file.name
    else:
        data_file = _state.configs["db_filename"]

    filename = _state.configs["db_filename"]
    if filename == "None":
        return

    file_pre, file_ext = os.path.splitext(filename)
    if file_ext == ".csv":
        _state.database = pd.read_csv(data_file, index_col=0)
    elif (file_ext == ".xlsx") or (file_ext == ".xls"):
        _state.database = pd.read_excel(data_file, index_col=0)

    create_pages()


def create_pages():
    """Add or change state parameters that are related to labeling pages

    These parameters assign the labels to multiple pages, with
    one label per page.
    """
    _state["pages"] = _state.database.columns[-_state.num_labels:]
    _state["classes"] = {label: sorted(list(_state.database[label].dropna().unique()))
                         for label in _state.pages}
    _state["num_to_class"] = {label: dict(enumerate(_state.classes[label]))
                              for label in _state.pages}
    _state["class_to_num"] = {label: {c: i for i, c in enumerate(_state.classes[label])}
                              for label in _state.pages}

    _state.update({
        'counter': 1,
        'local_results': {},
        'next_clicked': 0,
    })

    _state["next_"] = False
    _state["previous_"] = False

    _state["predictions"] = pd.DataFrame(index=_state.database.index,
                                         columns=_state.pages)

    file_pre, file_ext = os.path.splitext(_state.configs["db_filename"])
    try:
        with open(file_pre+str(_state.num_labels)+_MODEL, 'rb') as _file:
            _state["models_params"] = pkle.load(_file)
            _state["models"] = {}
            for label in _state.pages:
                _state.models[label] = ExplainableBoostingClassifier()
                _state.models[label].__dict__.update(_state.models_params[label])
    except FileNotFoundError:
        _state["models"], _state["models_params"] = initialize_models()

    compute_unlabeled_index()


def initialize_models():
    """initialize and train EBMs for all labels.

    If a pickle file of EBM models (stored in _MODEL) is not found
    in the directory, this function will be called by init_state_params()
    to initialize the models.
    """
    models = {}
    models_params = {}
    for label in _state.pages:
        y = _state.database[label].dropna().map(_state.class_to_num[label])
        X = subset_features(_state.database, label)
        X = X.loc[y.index, :]
        models[label] = ExplainableBoostingClassifier().fit(X, y)
        models_params[label] = models[label].__dict__

    return models, models_params


def subset_features(X, label):
    """Returns a subset of features specified in state's input_features parameters
    Args:
        X: a Pandas DataFrame.
        label: The column name of the labels.

    Returns
        A Pandas DataFrame consisting of a subset of features in X.
    """
    input_features = _state.configs["input_features"]
    if label not in input_features.keys():
        X = X.iloc[:, :-_state.num_labels]
    else:
        X = X.loc[:, input_features[label]]
    return X


def compute_unlabeled_index(new_labeled_index=None, label=None):
    """Track the indices of unlabeled data after introducing new labels.

    Args:
        new_labeled_index: A pandas index of newly labeled data.
        label: The column name of the new labels.
    """
    if new_labeled_index is not None:
        _state.unlabeled_index[label] = _state.unlabeled_index[label].difference(new_labeled_index)
    else:
        all_index = _state.database.index
        _state.unlabeled_index = {label: all_index[_state.database[label].isna()]
                                  for label in _state.pages}


def create_config_file():
    """Create a new config file"""
    _state["configs"] = {
        "db_filename": "None",
        "sidebar": {
            "num_labels": 1,
            "relabel": "Yes",
            "mode": "Fixed sample size",
            "n_samples": 50,
            "threshold": 0.95
        },
        "input_features": {}
    }
    with open(_CONFIGS_FILE, "w") as _file:
        json.dump(_state.configs, _file, indent=4)


def display_main_screen(label):
    """Display predictions and heatmaps on the main screen.

    This function is called after EBM has been trained on the labeled data.
    The predictions and explanations (displayed as heatmaps) will be shown
    on the main screen.

    Args:
        label: the column name of the predictions.
    """
    main_cols = st.columns((4, 4, 4))
    if _state.unlabeled_index[label].empty:
        main_cols[1].write("All "+label+" data are labeled.")
    else:
        with st.form("Label form"):
            if _state.local_results[label] == {}:
                main_cols[1].write("""There are some unlabeled data left.  \n \
                This means that the confidences of the remaining data are \
                above the threshold. \n You can either let the model label \
                these data automatically \n or change the sampling mode to \
                \"Fixed sample size\".""")
            else:
                input_features = _state.configs["input_features"]
                if label not in input_features.keys():
                    num_features = _state.database.shape[1] - _state.num_labels
                else:
                    num_features = len(input_features[label])
                num_heatmap_rows = math.ceil(num_features/_NUM_FEAT_PER_ROW)

                for page in _state.local_results[label]:
                    current_plot = plot_all_features(_state.local_results[label][page]['data'],
                                                     title=str(page),
                                                     height=50,
                                                     num_rows=num_heatmap_rows)
                    cols = st.columns((6, 1))
                    #with cols[0]:
                    #    if _state.text1 is not None:
                    #        st.write(_state.data[_state.text1][page])
                    #    if _state.text2 is not None:
                    #        st.write(_state.data[_state.text2][page])

                    cols[0].altair_chart(current_plot, use_container_width=True)

                    prediction = _state.local_results[label][page]['prediction']
                    cols[1].radio(
                        "Label",
                        options=_state.classes[label],
                        key=label+str(page),
                        index=int(prediction))
                    results = report_results(page, label)
                    for result in results:
                        cols[1].write(result)
                    st.markdown("""---""")

            label_from_cols = st.columns((4, 4, 4))

            label_from_cols[1].radio(
                "Automatically label the remaining data?",
                ("Yes", "No"),
                index=1,
                key="auto"
            )

            label_from_cols[1].form_submit_button("Submit Labels",
                                                  on_click=update_and_save,
                                                  args=(label,)
                                                  )

    button_cols = st.columns((3, 1, 1, 4))
    button_cols[1].button("Previous", on_click=update_previous_click)
    button_cols[2].button("Next", on_click=update_next_click)

    components.html(
        f"""
        <p>{_state.counter}</p>
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
     """,
        height=0
    )


@st.experimental_memo
def plot_all_features(data, title, height, num_rows):
    """Plot all rows of the heatmap of EBM's per-instance explanation.

    Args:
        data: Per-instance local explanations from EBM.
        title: The plot's title.
        height: The height of the plot.
        num_rows: The number of rows of the heatmap.

    Returns:
        obj: An Altair plot object.
    """
    plot_list = [None]*num_rows
    if num_rows == 1:
        plot_list[0] = plot(data,
                            title,
                            height)
    else:
        plot_list[0] = plot(data.iloc[0: _NUM_FEAT_PER_ROW],
                            title,
                            height)
        for i in range(1, num_rows-1):
            plot_list[i] = plot(data.iloc[_NUM_FEAT_PER_ROW*i: _NUM_FEAT_PER_ROW*(i+1)],
                                "",
                                height)
        plot_list[-1] = plot(data.iloc[_NUM_FEAT_PER_ROW*(num_rows-1):],
                             "",
                             height)

    obj = alt.vconcat(*plot_list).configure_axis(
        labelFontSize=13,
        titleFontSize=16,
        labelAngle=0,
        title=None
    ).configure_title(fontSize=16)

    return obj
                 

def plot(data, title, height):
    """Plot each row of the heatmap of EBM's per-instance explanation.

    Args:
        data: Per-instance local explanations from EBM.
        title: The plot's title.
        height: The height of the plot.

    Returns:
        obj: An Altair plot object.
    """
    base = alt.Chart(data).encode(
        x=alt.X('features', sort=None)
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color('scores:Q',
                        scale=alt.Scale(scheme='redblue', reverse=True, domain=[0,1]),
                        legend=alt.Legend(direction='vertical')
                        )
    )

    # Configure text
    text = base.mark_text(baseline='middle', fontSize=14).encode(
        text='values:N',
        color=alt.condition(
            (alt.datum.scores > 0.8) | (alt.datum.scores < 0.2),
            alt.value('white'),
            alt.value('black')
        )
    )

    obj = (heatmap+text).properties(height=height,
                                    width=650,
                                    title=title
                                    )

    return obj


@st.experimental_memo
def report_results(idx, col_name):
    """Create a list that contains current label (if exists) and confidence score.

    Args:
        idx: A row's index in the database.
        col_name: A column's name in the database.

    Returns:
        results: A list of current label (if exists) and confidence score.
    """
    results = []
    current_label = _state.database[col_name][idx]
    if not pd.isna(current_label):
        results.append(f"Current label: {current_label}")

    confidence = _state.local_results[col_name][idx]['confidence']
    results.append(f"Confidence: {confidence:.2f}")

    return results


def update_previous_click():
    """Track the index of the previous page."""
    _state.next_clicked -= 1
    if _state.next_clicked == -1:
        _state.next_clicked = len(_state.pages)-1
    _state.counter += 1


def update_next_click():
    """Track the index of the next page."""
    _state.next_clicked += 1
    if _state.next_clicked == len(_state.pages):
        _state.next_clicked = 0
    _state.counter += 1


def update_counter():
    """Update the counter after changing to a new screen."""
    if not (_state.next_ or _state.previous_):
        _state.next_clicked = _state.pages.get_loc(_state.label_page)
    _state.counter += 1


def sample_and_predict():
    """Sample data and make a dict of predictions and explanations.

    This function calls EBM to predict the labels and give per-instance
    local explanations. This function calls generate_explanation() to store
    the predictions and explanations in a dictionary.
    """
    st.experimental_memo.clear()

    if _state.loaded_new_file:
        init_state_params()
        _state.loaded_new_file = False
    else:
        if "database" not in _state:
            st.error("No database has been uploaded.")
            return
        if _state.configs["sidebar"]["num_labels"] != _state.num_labels:
            create_pages()

    _state.local_results = dict.fromkeys(_state.pages)

    for label in _state.pages:
        X = subset_features(_state.database, label)
        if _state.relabel == "No":
            X_unlabeled = X.loc[_state.unlabeled_index[label], :]
        else:
            X_unlabeled = X
        _state.local_results[label] = {}

        model = _state.models[label]
        generate_explanation(X_unlabeled, label, model)

    for k in _state.configs["sidebar"].keys():
        _state.configs["sidebar"][k] = _state[k]
    with open(_CONFIGS_FILE, "w") as _file:
        json.dump(_state.configs, _file, indent=4)


def update_and_save(label):
    """Update the labels, then retrain and save the models.

    Store the user's labels in the database, which is then saved to
    a local disk. EBM is then retrained on the database with addition
    labels, after which, a new list of predictions and explanations
    will be shown on the main screen. This function calls
    generate_explanation() to store the predictions and explanations
    in a dictionary.

    Args:
        label: the column name of the label.
    """
    new_labeled_index = list(_state.local_results[label].keys())
    _state.database.loc[new_labeled_index, label] = [_state[label+str(ix)]
                                                     for ix in new_labeled_index]
    compute_unlabeled_index(new_labeled_index, label)

    if _state.auto == "Yes":
        unlabeled_idx = _state.unlabeled_index[label]
        class_pred = _state.predictions.loc[unlabeled_idx, label]
        _state.database.loc[unlabeled_idx, label] = class_pred
        _state.unlabeled_index[label] = pd.Index([])
        labeled_index = _state.database.index
    else:
        labeled_index = _state.database.index.difference(_state.unlabeled_index[label])

    filename = _state.configs["db_filename"]
    file_pre, file_ext = os.path.splitext(filename)
    if file_ext == ".csv":
        _state.database.to_csv(filename)
    elif (file_ext == ".xlsx") or (file_ext == ".xls"):
        _state.database.to_excel(filename)

    X = subset_features(_state.database, label)
    X_train = X.loc[labeled_index, :]
    ytrain = _state.database.loc[labeled_index, label]
    ebm = ExplainableBoostingClassifier()
    ebm.fit(X_train, ytrain.map(_state.class_to_num[label]))
    _state.models[label] = ebm
    _state.models_params[label] = ebm.__dict__

    with open(file_pre+str(_state.num_labels)+_MODEL, 'wb') as _file:
        pkle.dump(_state.models_params, _file, protocol=pkle.HIGHEST_PROTOCOL)

    _state.local_results[label] = {}
    if _state.auto == "No":
        X = X.loc[_state.unlabeled_index[label], :]
        generate_explanation(X, label, ebm)

    _state.counter += 1


def generate_explanation(X, label, model):
    """Create a dict of predictions and explanations of a sample.

    Make label predictions and per-instance local explanations,
    which are then stored as a dict in _state.local_results.

    Args:
        X: A set of unlabeled data.
        label: The column name of a label.
        model: A model to predict labels and provide explanations.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    localx = model.explain_local(X)._internal_obj['specific']
    ypred = np.array([_state.num_to_class[label][localx[j]['perf']['predicted']]
                      for j in range(n_samples)])
    _state.predictions.loc[X.index, label] = ypred
    y = _state.database.loc[X.index, label]

    p = np.array([localx[j]['perf']['predicted_score'] for j in range(n_samples)])
    scores = np.minimum(p, (pd.isnull(y) | (ypred == y)))

    if _state.mode == "Confidence threshold":
        top_ind = np.where(scores <= _state.threshold)[0]
    else:
        n_samples = np.minimum(_state.n_samples, scores.shape[0]-1)
        top_ind = np.argpartition(scores, n_samples)[:n_samples]

    X_ = X.iloc[top_ind, :].copy()
    ypred = ypred[top_ind]

    id_idx_pair = dict(zip(X_.index, top_ind))

    try:
        data_by_class = [X_[ypred == c] for c in _state.classes[label]]
    except KeyError:
        return

    feature_names = X.columns

    for sgn_data in data_by_class:
        current_dict = _state.local_results[label]
        for j in sgn_data.index:
            localxi = localx[id_idx_pair[j]]

            if len(_state.classes[label]) == 2:
                feature_contrib = localxi['scores'][:n_features]
            else:
                feature_contrib = [localxi['scores'][k][localxi['perf']['predicted']]
                                   for k in range(n_features)]
            heatmap_data = pd.DataFrame({'features': feature_names,
                                         'values': localxi['values'][:n_features],
                                         'scores': 1/(1+1/np.exp(feature_contrib))})
            heatmap_data = heatmap_data.astype({'features': str,
                                                'values': str,
                                                'scores': float})
            current_dict[j] = {
                'actual': localxi['perf']['actual'],
                'prediction': localxi['perf']['predicted'],
                'confidence': localxi['perf']['predicted_score'],
                'data': heatmap_data}


if __name__ == "__main__":
    main()
