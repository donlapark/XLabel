# XLabel: e**X**plainable **Label**ing Assistant

[![License](https://img.shields.io/github/license/donlapark/XLabel)](LICENSE)
![Python](https://img.shields.io/badge/python-3.7_|_3.8-blue.svg)
![LGTM Grade](https://img.shields.io/lgtm/grade/python/github/donlapark/XLabel)
[![CodeQL](https://github.com/donlapark/XLabel/actions/workflows/codeql.yml/badge.svg)](https://github.com/donlapark/XLabel/actions/workflows/codeql.yml)
![Maintenance](https://img.shields.io/maintenance/yes/2022)

XLabel is an open-source [Streamlit](https://streamlit.io/) app that takes an explainable machine learning approach to visual-interactive data labeling.

Try out the app at [<ins>Hugging Face Demo</ins>](https://huggingface.co/spaces/Donlapark/XLabel). No need to upload data!

## Features
XLabel can:
* Predict the most probable labels using Explainable Boosting Machine (EBM) [[2]](#2).
* Show the contributions of each feature towards the predicted labels.
* Provide an option to write the labels directly into the data file (use `XLabel.py`) or save them in a separate file (use `XLabelDL.py`)
* Support data with multiple labels and multiple classes.
* Support data with missing values ([thanks to EBM](https://github.com/interpretml/interpret/issues/18)) and/or non-numeric categorical features.

![Screenshot](screenshot/XLabel_screenshot.png)

## Installation
All you need is installing the required packages, namely [Streamlit](https://streamlit.io/) and [InterpretML](https://github.com/interpretml/interpret).
```
python3 -m pip install streamlit interpret
```
If you are using Anaconda, first activate an environment and run the following line:
```
conda install streamlit interpret
```

## Usage
Before using XLabel, the data file must follow the following tabular convention:
* The file must be in either CSV or Excel format.
* The first row of the file must be the names of the columns.
* The first column must contain a unique identifier (id) for each row.
* The label columns must appear last.
In addition, a few instances must have already been labeled, with each class appearing at least once (For example, if a label has five possible classes, then the required minimum number of labeled instances is 5).

With your data file satisfying these conditions, you can now start data labeling with XLabel!
1. Copy `XLabel.py` to the directory that contains the data file and run the `streamlit` command:
    ```
    streamlit run XLabel.py
    ```
    * By design, `XLabel.py` will write the labeled data to the original data file. If instead you would like to download the labeled data as a separate file, use `XLabelDL.py` instead.
    * You can assign a specific list of input features for each label by editing `configs.json` and copying it along with `XLabel.py`. There are also other sidebar options that you can play around as well. Here is an example ofr [`configs.json`](configs.json).
2. Upload a data file (only on the first run), select the options on the sidebar, and then click "**Sample**". The samples with lowest predictive confidences will be shown first in the main screen.
3. Check the suggested labels; you can keep the correct ones and change the wrong ones.
4. Click the "**Submit Labels**" button at the bottom of the page to save the labels. 
    * If you are using `XLabel.py`, the labels will be saved directly to the original data file.
    * If you are using `XLabelDL.py`, you need to click the `Download labeled data` in the sidebar to download the labeled data as a new file.

## Citing this work
If you use XLabel in your work, please cite:

<a id="1">[1]</a> 
> Ponnoprat, D., Pattarapanitchai, P., Taninpong, P., & Suantai, S. (2022). An Explainable Machine Learning Approach to Visual-Interactive Labeling: A Case Study on Non-communicable Disease Data. doi:10.48550/arXiv.2209.12778

    @misc{https://doi.org/10.48550/arxiv.2209.12778,
    doi = {10.48550/arXiv.2209.12778},
    url = {https://arxiv.org/abs/2209.12778},
    author = {Ponnoprat, Donlapark and Pattarapanitchai, Parichart and Taninpong, Phimphaka and Suantai, Suthep},
    keywords = {Machine Learning (cs.LG), Human-Computer Interaction (cs.HC), Applications (stat.AP), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {An Explainable Machine Learning Approach to Visual-Interactive Labeling: A Case Study on Non-communicable Disease Data},
    publisher = {arXiv},
    year = {2022},
    }

<a id="2">[2]
></a> "InterpretML: A Unified Framework for Machine Learning Interpretability" (H. Nori, S. Jenkins, P. Koch, and R. Caruana 2019)


    @article{nori2019interpretml,
      title={InterpretML: A Unified Framework for Machine Learning Interpretability},
      author={Nori, Harsha and Jenkins, Samuel and Koch, Paul and Caruana, Rich},
      journal={arXiv preprint arXiv:1909.09223},
      year={2019}
    }    

