'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/blueprints/supervised.py
'''

from distutils.util import strtobool
import json, os
from flask import (current_app,
                   request,
                   Blueprint)

from famie.api.api_fns.project_settings.supervised import set_class_names, parse_labeled_data
from famie.api.blueprints.common import al_controllers, config
from famie.api.active_learning.utils import *
from famie.constants import PROJECT_TYPE_CLASSIFICATION, PROJECT_TYPE_NER
import time

supervised_bp = Blueprint('supervised', __name__)


def check_supervised_project(project):
    if not project.type in [PROJECT_TYPE_CLASSIFICATION, PROJECT_TYPE_NER]:
        raise Exception(f"Project type {project.type} is not supervised, calling the "
                        "wrong API.")


@supervised_bp.route('/api/classnames', methods=['POST'])
def set_classnames():
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    column_name = request.form['column_name']
    if not column_name:
        raise Exception("Column name undefined")

    file = request.files['file']
    if not file:
        raise Exception("File is undefined")

    class_names = set_class_names(file)
    ensure_dir(os.path.join(DATABASE_DIR, project_name))
    with open(os.path.join(DATABASE_DIR, project_name, 'vocabs.json'), 'w') as f:
        json.dump({x['name']: x['id'] for x in class_names}, f, ensure_ascii=False)

    project_info = get_project_info(project_name)
    if project_info:
        project_info['classes'] = class_names
        update_project_to_database(project_info)

    return json.dumps(class_names)


@supervised_bp.route('/api/start-iteration', methods=['POST'])
def start_iteration():
    print('-' * 50)
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    update_id = request.form['update_id']
    if not update_id:
        raise Exception("Need update id for polling")

    # proxy model retraining & selection
    annotated, unlabeled = get_examples(project_name)

    if len(annotated) > 0 or os.path.exists(os.path.join(DATABASE_DIR, project_name, 'provided-labeled-data.json')):
        project_info = get_project_info(project_name)
        project_type = project_info['type']

        # make the active learning controller be listening if it is not
        active_task = al_controllers['active_task']
        if active_task != project_type or project_name != al_controllers['active_project']:
            al_controllers[active_task].stop_listening()

        if os.path.exists(os.path.join(DATABASE_DIR, project_name, 'provided-labeled-data.json')):
            with open(os.path.join(DATABASE_DIR, project_name, 'provided-labeled-data.json')) as f:
                provided_labeled_data = [x['example_id'] for x in json.load(f)]
        else:
            provided_labeled_data = []

        al_controllers[project_type].listen(project_state={
            'project_dir': os.path.join(DATABASE_DIR, project_name),
            'project_id': project_name,
            'annotations': annotated,
            'provided_labeled_data': provided_labeled_data
        })

        al_controllers['active_task'] = project_type
        al_controllers['active_project'] = project_name

        al_controllers[project_type].run_proxy_model(unlabeled, project_name)
        while True:
            time.sleep(LISTEN_TIME)
            if al_controllers[project_type].trainer['proxy'].signal != RUN_PROXY:
                break
        with open(os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json')) as f:
            selected_unlabeled = [json.loads(line.strip()) for line in f if line.strip()]

        if len(annotated) == 0:
            al_controllers[project_type].proxy_model_predicts(selected_unlabeled, project_name)
            while True:
                time.sleep(LISTEN_TIME)
                if al_controllers[project_type].trainer['proxy'].signal != PROXY_PREDICTS:
                    break
        else:
            al_controllers[project_type].target_model_predicts(selected_unlabeled, project_name)
            while True:
                time.sleep(LISTEN_TIME)
                if al_controllers[project_type].trainer['target'].signal != TARGET_PREDICTS:
                    break

    else:
        for keyname in SUPPORTED_TASKS:
            al_controllers[keyname].stop_listening()

        selected_unlabeled = random_select(unlabeled, selected_size=config.num_examples_per_iter)

        with open(os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json'), 'w') as f:
            for d in selected_unlabeled:
                f.write(json.dumps(d) + '\n')

    # show suggested examples for the next iteration
    project_stats = get_project_stats(project_name)

    assert len(annotated) == project_stats['docs']['total_manual_docs']
    project_stats['docs']['total_documents'] = len(annotated) + len(selected_unlabeled)

    project_stats['update_id'] = update_id
    project_stats['docs']['update_id'] = update_id
    return json.dumps(project_stats)


@supervised_bp.route('/api/get-docs', methods=['GET'])
def get_docs():  # start the annotation and the training of the target model here
    print('-' * 50)
    project_name = request.args['project_name']

    if not os.path.exists(os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json')):
        annotated, unlabeled = get_examples(project_name)
        selected_unlabeled = random_select(unlabeled, selected_size=10)
        with open(os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json'), 'w') as f:
            for d in selected_unlabeled:
                f.write(json.dumps(d) + '\n')

    if os.path.exists(os.path.join(DATABASE_DIR, project_name, 'annotations.json')) or os.path.exists(os.path.join(DATABASE_DIR, project_name, 'provided-labeled-data.json')):
        project_info = get_project_info(project_name)
        project_type = project_info['type']
        if al_controllers[project_type].is_listening and al_controllers['active_project'] == project_name:
            # run the target model
            al_controllers[project_type].run_target_model(project_name)
    else:
        for keyname in SUPPORTED_TASKS:
            al_controllers[keyname].stop_listening()

    with open(os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json')) as f:
        selected_unlabeled = [json.loads(line.strip()) for line in f if line.strip()]

    res = {
        'total': len(selected_unlabeled),
        'docs': selected_unlabeled,
        'labels': None,
        'doc_ids': [ul['example_id'] for ul in selected_unlabeled]
    }
    return json.dumps(res)


@supervised_bp.route('/api/label-entity', methods=['POST'])
def label_entity():  # each time you finish the annotation for each sentence by clicking the "V" button -> this function is called
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    doc_id = request.form['doc_id']
    try:
        doc_id = int(doc_id)
    except:
        raise Exception("Doc id undefined")

    try:
        spans = check_entity_form_input(json.loads(request.form['spans']))  # annotations for one sentence
    except:
        raise Exception("Manual label undefined or badly formed.")

    session_id = request.form['session_id']
    if not session_id:
        raise Exception("Session id undefined or badly formed.")

    if not os.path.exists(os.path.join(DATABASE_DIR, project_name, 'annotations.json')):
        with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json'), 'w') as f:
            f.write('')

    with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json'), 'a') as f:
        f.write(json.dumps({
            'example_id': doc_id,
            'spans': spans
        }) + '\n')
    return json.dumps(spans)


@supervised_bp.route('/api/export-rules', methods=['POST'])
def export_rules():
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    saved_fpath = os.path.join(OUTPUT_DIR, project_name, 'target_output_weights.ckpt')
    if not os.path.exists(saved_fpath):
        print('{} does not exist!'.format(saved_fpath))
        return json.dumps({})

    with open(saved_fpath) as f:
        json_ckpt = f.read()

    return json_ckpt


@supervised_bp.route('/api/download-unlabeled', methods=['POST'])
def download_unlabeled():
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    data_fpath = os.path.join(DATABASE_DIR, project_name, 'selected-unlabeled-data.json')
    if not os.path.exists(data_fpath):
        print('{} does not exist!'.format(data_fpath))
        return json.dumps({})

    with open(data_fpath) as f:
        selected_unlabeled = [json.loads(line.strip()) for line in f if line.strip()]
        outputs = [{
            'example_id': dpoint['example_id'],
            'text': dpoint['text'],
            'tokens': [t['text'] for t in dpoint['tokens']]
        } for dpoint in selected_unlabeled]

    return json.dumps({'project_name': project_name, 'data': outputs})


@supervised_bp.route('/api/upload-labeled-data', methods=['POST'])
def upload_labeled_data():
    project_name = request.form['project_name']
    if not project_name:
        raise Exception("Project name undefined")

    column_name = request.form['column_name']
    if not column_name:
        raise Exception("Column name undefined")

    file = request.files['file']
    if not file:
        raise Exception("File is undefined")

    provided_labeled_data = parse_labeled_data(file)
    ensure_dir(os.path.join(DATABASE_DIR, project_name))
    ensure_dir(os.path.join(DATABASE_DIR, project_name, 'trankit_features'))
    with open(os.path.join(DATABASE_DIR, project_name, 'provided-labeled-data.json'), 'w') as f:
        json.dump([{'example_id': d['example_id']} for d in provided_labeled_data], f, ensure_ascii=False)

    progress = tqdm.tqdm(total=len(provided_labeled_data), ncols=75,
                         desc='Preprocessing labeled data')

    for d in provided_labeled_data:
        progress.update(1)

        tokens = d['tokens']
        ######### subword tokenization #######
        proxy_piece_idxs, proxy_attn_masks, proxy_token_lens = subword_tokenize(
            tokens, config.proxy_tokenizer,
            config.max_sent_length
        )
        target_piece_idxs, target_attn_masks, target_token_lens = subword_tokenize(
            tokens, config.target_tokenizer,
            config.max_sent_length
        )
        ######### read annotations ###########
        labels = d['labels']

        inst = {
            'example_id': d['example_id'],
            'text': 'original text is not provided',
            'tokens': tokens,

            'proxy_piece_idxs': proxy_piece_idxs,
            'proxy_attention_mask': proxy_attn_masks,
            'proxy_token_lens': proxy_token_lens,

            'target_piece_idxs': target_piece_idxs,
            'target_attention_mask': target_attn_masks,
            'target_token_lens': target_token_lens,

            'labels': labels
        }
        with open(os.path.join(DATABASE_DIR, project_name, 'trankit_features', '{}.json'.format(d['example_id'])), 'w') as f:
            json.dump(inst, f, ensure_ascii=False)

    progress.close()

    return json.dumps({})
