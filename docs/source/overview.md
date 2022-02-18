# Quick examples

## Initialization
To start an annotation session, please use the following command:
```python
famie start
```
This will run a server on users' local machines (no data or models will leave users' local machines), users can access FAMIE's web interface via the URL: http://127.0.0.1:9000/

To start a new project, users need to upload an unlabeled dataset file with an entity type file (in text format) to the web interface. After that, they will be directed to a data statistic page. Clicking on the bottom left corner will start the labeling process.

<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/0_newproj.png" height="300px"/></div>

## Annotation

for each data sample, annotators first select a label from dropdown, then proceed to highlight appropriate spans for the corresponding labels.
<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/1_select_label.png" height="300px"/></div>
<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/2_anno_span.png" height="300px"/></div>

Annotators continue labeling until all entities in the given sentence are covered, from which they can proceed by clicking save button and then next arrow to go to the next example.
<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/3_save_next.png" height="300px"/></div>

After finishing labeled every unlabeled data of the current iteration, clicking on **Finish Iter** will take users to a waiting page for the next iteration (during this time, the proxy model is being retrained with the new labeled data, which usually takes about 3 to 5 minutes).
<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/4_fin_prox.png" height="300px"/></div>

## Output
 FAMIE allows users to download the trained models and annotated data of the current round via the web interface.
<div align="center"><img src="https://raw.githubusercontent.com/nlp-uoregon/famie/main/pics/download.png" height="300px"/></div>

FAMIE also provides a simple and intuitive code
interface for interacting with the resulting labeled
dataset and trained main models after the AL processes.

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

# use the trained model to make predictions
model.predict('Oregon is a beautiful state!')
# ['B-Location', 'O', 'O', 'O', 'O']