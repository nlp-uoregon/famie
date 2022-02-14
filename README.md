<h2 align="center">FAMIE: A Fast Active Learning Framework for Multilingual Information Extraction</h2>

<div align="center">
    <a href="https://github.com/nlp-uoregon/famie/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/nlp-uoregon/famie.svg?color=blue">
    </a>
    <a href='https://famie.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/famie/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="http://nlp.uoregon.edu:9000/">
        <img alt="Demo Website" src="https://img.shields.io/website/http/famie.readthedocs.io/en/latest/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://pypi.org/project/famie/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/famie?color=blue">
    </a>
    <a href="https://pypi.org/project/famie/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/famie?colorB=blue">
    </a>
</div>

FAMIE is a comprehensive  and efficient [active learning]() (AL) toolkit for multilingual information extraction (IE). FAMIE is designed to address a fundamental problem in existing AL frameworks where annotators need to wait for a long time between annotation batches due to the time-consuming nature of model training and data selection at each AL iteration. With a novel proxy AL mechanism and the integration of our SOTA multilingual toolkit [Trankit](https://github.com/nlp-uoregon/trankit), FAMIE can quickly provide users with a labeled dataset and a ready-to-use model for different IE tasks over [100 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages).

FAMIE's documentation page: https://famie.readthedocs.io

FAMIE's demo website: http://nlp.uoregon.edu:9000/


### Installation
FAMIE can be easily installed via one of the following methods:
#### Using pip
```
pip install famie
```
The command would install FAMIE and all dependent packages automatically. 

#### From source
```
git clone https://github.com/nlp-uoregon/famie.git
cd famie
pip install -e .
```
This would first clone our github repo and install FAMIE.

### Usage
FAMIE currently supports Named Entity Recognition and Event Detection for over [100 languages](https://trankit.readthedocs.io/en/latest/pkgnames.html#trainable-languages). Using FAMIE includes three following steps:
- Start an annotation session.
- Annotate data for a target task.
- Access the labeled data and a ready-to-use model returned by FAMIE.

#### Starting an annotation session
To start an annotation session, please use the following command:
```python
famie start
```
This will run a server on users' local machines (no data or models will leave users' local machines), users can access FAMIE's web interface via the URL: http://127.0.0.1:9000/
. As FAMIE is an AL framework, it provides different data selection algorithms that recommend users the most beneficial examples to label at each annotation iteration. This is done via passing an optional argument `--selection [mnlp|badge|bertkm|random]`.

#### Annotating data

#### Accessing the labeled data and the trained model
```python
import famie

# access a project via its name
p = famie.get_project('named-entity-recognition') 

# access the project's labeled data
data = p.get_labeled_data() # a Python dictionary

# export the project's labeled data to a file
p.export_labeled_data('data.json')

# export the project's trained model to a file
p.export_trained_model('model.ckpt')

# access the project's trained model
model = p.get_trained_model()

# access a trained model from file
model = famie.load_model_from_file('model.ckpt')

# use the trained model to make predicions
model.predict('Oregon is a beautiful state!')
# ['B-Location', 'O', 'O', 'O', 'O']
```
