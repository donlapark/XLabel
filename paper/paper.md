---
title: 'XLabel: An Explainable Machine Learning Approach to Visual-Interactive Labeling'
tags:
  - human-in-the-loop
  - explainable artificial intelligence
  - interpretable
  - interactive labeling
authors:
  - name: Donlapark Ponnoprat
    corresponding: true
    affiliation: 1
  - name: Parichart Pattarapanitchai
    affiliation: 1
  - name: Phimphaka Taninpong
    affiliation: 1
  - name: Suthep Suantai
    affiliation: 2
affiliations:
 - name: Department of Statistics, Chiang Mai University, Chiang Mai, Thailand
   index: 1
 - name: Department of Mathematics, Chiang Mai University, Chiang Mai, Thailand
   index: 2
date: XX
bibliography: paper.bib
---

# Summary

We introduce a new visual-interactive tool: Explainable Labeling Assistant (`XLabel`) that takes an explainable machine learning approach to data labeling. The main component of `XLabel` is the Explainable Boosting Machine (EBM), a predictive model that can calculate the contribution of each input feature towards the final prediction. This model provides visual explanations to assist users in understanding and verifying predictions, making it an invaluable tool for data labeling. We can reduce bias and increase transparency in the data labeling process by using an explainable ML model. The detailed code, documentation, and installation instructions of `XLabel` have been made available at: https://github.com/donlapark/XLabel. 

# Statement of need

Labeling massive amounts of data can be very time-consuming in some industries and require significant human effort, such as labeling a medical patient's record: as positive or negative. To reduce experts' workload, we design a visual-interactive tool called Explainable Labeling Assistant (`XLabel`). Most important part of `XLabel` is a prediction model that takes a patient's record as an input, and suggests a label $y\in\{0,1\}$ of that record to the user. 

To ensure that the model's suggestions are trustworthy, we take an explainable approach; the model must be able to explain the reasons behind its suggestions. The labeling process with `XLabel` is the following :

- `XLabel` reads the data of all unlabeled records. For each record, it creates a *pseudo-label* $\hat y \in \{0,1\}$. 
- `XLabel` samples a subset of records; it then presents the records, their pseudo-labels and the explanations to the user.
- The user reads the explanations, then turns the pseudo-labels into true labels by keeping the correct pseudo-labels and flipping the wrong ones (i.e., from $0$ to $1$ or $1$ to $0$).
- `XLabel` accepts the labels from the user and retrain its prediction model. Now the model can provide more accurate pseudo-labels to the next unlabeled sample.

The user's labeling workload will be vastly reduced if most of the pseudo-labels are already correct. Thus in addition to being explainable, the prediction model inside `XLabel` must be accurate. Recently, there have been a series of work that show that, contrary to popular belief that there is a trade-off between explainability and accuracy, it is possible for a machine learning model to be both explainable and accurate.

### Explainable Boosting Machine (EBM)

The prediction model that we use in `XLabel` is *Explainable Boosting Machine* (EBM) [@Lou2013; @nori2019], an explainable version of gradient boosting machine [@Friedman2001; @Friedman2002], which is known for its predictive performance.

Let $x=(x_1,\ldots,x_n)$ be a patient's record with true label $y\in\{0,1\}$. The EBM is an additive model, that is, its prediction on $x$ is given by

\begin{equation}
    f(x) = \beta_0 + \sum_i f_i(x_i) + \sum_{i,j} f_{ij}(x_i,x_j),
\end{equation}

where $\beta_0$ is the intercept, and each $f_i$ is a sum of regression trees, that is, 

\begin{equation} \label{eq:sum}
    f_i(x_i) = \sum_{k} f_{ik}(x_i)\\
    f_{ij}(x_i,x_j) = \sum_{k} f_{ijk}(x_i,x_j).
\end{equation}

Here, $f_{ik}$ and $f_{ijk}$ are regression trees for all $i$, $j$ and $k$. 

The model then outputs the probability prediction through the logistic function:

\begin{equation}
    p_x = \Pr(y=1 \mid x) = \frac{1}{1+e^{-f(x)}}.
\end{equation}

The predicted label is then $\hat y = 1$ if $\Pr(y=1 \mid x)\geq 0.5$ and $\hat y = 0$ if $\Pr(y=1 \mid x)< 0.5$. 

### `XLabel` ’s Explanations

The fact that EBM is an additive model allows `XLabel` to measure the contribution from each feature towards the prediction.To be more precise, from \autoref{eq:sum}, we can treat $f_i(x_i)$ as the contribution from $x_i$ (we shall ignore the interactive terms $f_{ij}(x_i,x_j)$ as they are used to model the residual [@Lou2013]). In particular, $f(x_i)>0$ implies that $x_i$ contributes to a positive label, while $f(x_i)<0$ implies that $x_i$ contributes to a negative label.

To visualize these features' contributions, we choose *heatmap*; Its compact representation allows the user to scroll through the records very quickly. In each heatmap, a rectangle is drawn for each feature, and the color is determined by its contribution. Since the contribution of each variable can be an arbitrarily large positive or negative number, we have to scale it to a range of $(0,1)$ using the logistic function:
 
The rectangle is then colored red if $HEAT(xi)$ is close to 1 and blue if it is close to 0. This heatmap allows the user to quickly notice the features that contribute the most to the label, andthen promptly decide to keep or flip the label. 

\begin{equation}
    \operatorname{HEAT}(x_i) = \frac{1}{1+e^{-f_i(x_i)}}.
\end{equation}

# Acknowledgements

This research project was supported by Fundamental Fund 2022, Chiang Mai University undergrant number FF65/059. We thank Sriphat Medical Center, Chiang Mai, Thailand for providing valuable data. We also thank the physician team for labeling and validating data accuracy. 

# References
