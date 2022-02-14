'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/blueprints/common.py
'''
from datetime import datetime
from distutils.util import strtobool
import json

from flask import (current_app,
                   render_template,
                   request,
                   Blueprint)

from famie.api.api_fns.project_creation.common import get_upload_key

from famie.api.api_fns.project_creation.supervised import create_supervised_project

from famie.constants import (ALL_PROJECT_TYPES,
                              PROJECT_TYPE_CLASSIFICATION,
                              PROJECT_TYPE_ED,
                              PROJECT_TYPE_NER)
from famie.api.active_learning.controllers import *
from famie.api.active_learning.config import config

bp = Blueprint('api', __name__)

al_controllers = {'active_task': 'ner', 'active_project': 'asdf0qweo23904123ieaewklf'}
for supported_task in SUPPORTED_TASKS:
    al_controllers[supported_task] = Controller(config, task=supported_task)


@bp.route("/")
def index():
    return render_template('index.html')


@bp.route('/', defaults={'path': ''})
@bp.route('/<path:path>')
def catch_all(path):
    return render_template('index.html')


@bp.route('/hello')
def hello_world():
    return 'Hello, World!'


@bp.route('/api/upload', methods=['POST'])
def upload():
    start_time = datetime.now()
    file = request.files['file']
    if not file:
        raise Exception("File is undefined")

    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    try:
        column_name_mapping = json.loads(request.form['column_names'])
    except:
        raise Exception(f"Column name mapping not passed correctly")
    if not column_name_mapping:
        raise Exception("Column names undefined")

    project_type = request.form['project_type']
    if not project_type:
        raise Exception("Project type undefined")

    try:
        file_type = request.form['file_type']
        upload_key = get_upload_key(project_type, file_type)
    except:
        raise Exception(f"File type undefined or incorrect")

    upload_id = request.form[upload_key]
    if not upload_id:
        raise Exception("Need upload id for polling")

    if project_type not in ALL_PROJECT_TYPES:
        raise Exception(f"Non-recognised project type {project_type}. "
                        f"Needs to be one of: {ALL_PROJECT_TYPES}")

    class_names = None

    input_params = (config,
                    detect_lang,
                    file,
                    project_name,
                    project_type,
                    upload_id,
                    file_type)

    upload_folder = os.path.join(DATABASE_DIR, project_name)
    ensure_dir(upload_folder)
    project_id = create_supervised_project(*input_params, upload_folder)

    update_project_to_database(project_info={
        'id': project_id, 'name': project_name, 'type': project_type, 'classes': [], 'index_name': project_name, 'filename': os.path.join(upload_folder, 'unlabeled-data.json')
    })

    if project_id:
        end_time = datetime.now()
        print(f"Uploading took {(end_time - start_time).seconds / 60} minutes.")
    print('Uploading... done!')
    return json.dumps({"id": project_id, "class_names": class_names})


@bp.route('/api/delete-project/<project_name>', methods=['DELETE'])
def delete_project(project_name):
    if not project_name:
        raise Exception("Project name undefined")

    return "success"


@bp.route('/api/get-projects', methods=['GET'])
def get_projects():
    # stop all controllers when users see the list of projects
    for task in SUPPORTED_TASKS:
        al_controllers[task].stop_listening()

    project_list = get_project_list()
    return json.dumps(project_list)


@bp.route('/api/export-labels', methods=['POST'])
def export_labels_api():
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    labeled_fpath = os.path.join(OUTPUT_DIR, project_name, 'labeled-data.json')
    if not os.path.exists(labeled_fpath):
        print('{} does not exist!'.format(labeled_fpath))
        return ''
    else:
        with open(labeled_fpath) as f:
            data = json.load(f)
        return json.dumps({'project_name': project_name, 'data': data})

