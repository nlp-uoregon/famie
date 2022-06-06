'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/api_fns/project_creation/supervised.py
'''

import os
import json
import sys
import shutil
import traceback
import tqdm
from werkzeug.utils import secure_filename

from famie.api.api_fns.project_creation.common import (ES_indexer,
                                                       get_random_index_name,
                                                       UploadedFile)
from famie.constants import MAPPINGS

ALLOWED_EXTENSIONS = {'csv', 'txt'}


def get_anchors_from_tag_sequence(path):
    mentions = []
    cur_mention = None
    for j, tag in enumerate(path):
        if tag.startswith('B-') or tag.startswith('I-'):
            prefix = tag.split('-')[0]
            tag = tag[2:]
        else:
            prefix = tag = 'O'

        if prefix == 'B':
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = [j, j + 1, tag]
        elif prefix == 'I':
            if cur_mention is None:
                # treat it as B-*
                cur_mention = [j, j + 1, tag]
            elif cur_mention[-1] == tag:
                cur_mention[1] = j + 1
            else:
                # treat it as B-*
                mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
        else:
            if cur_mention:
                mentions.append(cur_mention)
            cur_mention = None
    if cur_mention:
        mentions.append(cur_mention)
    mentions = [(i, j, tag) for i, j, tag in mentions]
    return mentions


class UploadedSupervisedFile(UploadedFile):

    def __init__(self, project_type, input_data, file_type):
        super().__init__(project_type, input_data, file_type)

    def process_file(self, config, detect_lang_fn, project_name, project_full_path):
        if not os.path.exists(project_full_path):
            os.makedirs(project_full_path)

        json_writer = open(os.path.join(project_full_path, 'unlabeled-data.raw.json'), 'w')

        lid = 0
        for line in self:
            json_writer.write(json.dumps({
                'project_name': project_name,
                'example_id': lid,
                'text': line['text']
            }) + '\n')
            lid += 1

        json_writer.close()

        # determine whether the uploaded data is in plain text or JSON format

        try:
            uploaded_data = []
            with open(os.path.join(project_full_path, 'unlabeled-data.raw.json')) as f:
                for line in f:
                    if not line.strip():
                        continue

                    line = json.loads(line.strip())
                    text = line['text'].strip()
                    if text:
                        d = json.loads(text)
                        format_check = type(d) == dict and 'text' in d and 'tokens' in d and len(
                            d['tokens']) > 0 and 'text' in d['tokens'][0]

                        if format_check:
                            d['project_name'] = project_name
                            uploaded_data.append(d)

            uploaded_format = 'json'
            print('Uploaded data is in JSON format!')
        except json.decoder.JSONDecodeError:
            uploaded_format = 'plain-text'
            print('Uploaded data is in plain text format')

        project_task_type = 'unconditional'

        if uploaded_format == 'plain-text':
            ##################### lang detection ########################
            langid_inputs = []
            with open(os.path.join(project_full_path, 'unlabeled-data.raw.json')) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        langid_inputs.append(d['text'])
                        if len(langid_inputs) == 100:
                            break

            lang = detect_lang_fn(text='\n'.join(langid_inputs))

            if lang != config.trankit_tokenizer.active_lang:
                if lang not in config.trankit_tokenizer.added_langs:
                    config.trankit_tokenizer.add(lang)
                config.trankit_tokenizer.set_active(lang)

            print('Using {} tokenizer to preprocess unlabeled data ...'.format(lang))

            ####### tokenization and other trankit computations ########
            progress = tqdm.tqdm(total=lid, ncols=75,
                                 desc='Preprocessing')

            numberized_data = []

            with open(os.path.join(project_full_path, 'unlabeled-data.raw.json')) as f:
                for line in f:
                    line = line.strip()
                    progress.update(1)
                    if line:
                        d = json.loads(line)

                        trankit_tokens = config.trankit_tokenizer.tokenize(d['text'], is_sent=True)['tokens']
                        tokens = [t['text'] for t in trankit_tokens]
                        s_pieces = [[p for p in config.proxy_tokenizer.tokenize(w) if p != '▁'] for w in tokens]
                        for ps in s_pieces:
                            if len(ps) == 0:
                                ps += ['-']
                        s_token_lens = [len(ps) for ps in s_pieces]
                        s_pieces = [p for ps in s_pieces for p in ps]
                        # Pad word pieces with special tokens
                        piece_idxs = config.proxy_tokenizer.encode(
                            s_pieces,
                            add_special_tokens=True,
                            max_length=510,
                            truncation=True
                        )
                        if len(piece_idxs) > 510 or len(piece_idxs) == 0:
                            continue
                        attn_masks = [1] * len(piece_idxs)
                        ######### read annotations ###########
                        labels = ['O'] * len(tokens)
                        label_idxs = [0 for label in labels]
                        inst = {
                            'project_name': d['project_name'],
                            'project_task_type': 'unconditional',
                            'example_id': d['example_id'],
                            'text': d['text'],
                            'lang': lang,
                            'tokens': trankit_tokens,
                            'anchor': -1,  # no anchor information
                            'anchor_type': 'unknown',  # no anchor information
                            'piece_idxs': piece_idxs,
                            'attention_mask': attn_masks,
                            'token_lens': s_token_lens,
                            'labels': labels,
                            'label_idxs': label_idxs
                        }
                        numberized_data.append(inst)
            progress.close()
            with open(os.path.join(project_full_path, 'unlabeled-data.json'), 'w') as f:
                for nd in numberized_data:
                    f.write(json.dumps(nd) + '\n')

        else:
            assert uploaded_format == 'json'
            print('Uploaded data is already tokenized!')
            langid_inputs = [' '.join([t['text'] for t in d['tokens']]) for d in uploaded_data[:100]]
            lang = detect_lang_fn(text='\n'.join(langid_inputs))
            print('Start preprocessing ...')

            project_task_type = 'conditional'

            ####### tokenization and other trankit computations ########
            progress = tqdm.tqdm(total=len(uploaded_data), ncols=75,
                                 desc='Preprocessing')

            numberized_data = []
            for d in uploaded_data:
                progress.update(1)
                assert 'labels' in d

                tokens = [t['text'] for t in d['tokens']]
                s_pieces = [[p for p in config.proxy_tokenizer.tokenize(w) if p != '▁'] for w in tokens]
                for ps in s_pieces:
                    if len(ps) == 0:
                        ps += ['-']
                s_token_lens = [len(ps) for ps in s_pieces]
                s_pieces = [p for ps in s_pieces for p in ps]
                # Pad word pieces with special tokens
                piece_idxs = config.proxy_tokenizer.encode(
                    s_pieces,
                    add_special_tokens=True,
                    max_length=510,
                    truncation=True
                )
                if len(piece_idxs) > 510 or len(piece_idxs) == 0:
                    continue
                attn_masks = [1] * len(piece_idxs)

                anchors = get_anchors_from_tag_sequence(d['labels'])
                for anchor in anchors:
                    start_token, end_token, anchor_type = anchor
                    ######### read annotations ###########
                    labels = ['O'] * len(tokens)
                    label_idxs = [0 for label in labels]
                    inst = {
                        'project_name': d['project_name'],
                        'project_task_type': 'conditional',
                        'example_id': len(numberized_data),
                        'text': d['text'],
                        'tokens': d['tokens'],
                        'anchor': start_token,  # token position of the anchor
                        'anchor_type': anchor_type,
                        'piece_idxs': piece_idxs,
                        'attention_mask': attn_masks,
                        'token_lens': s_token_lens,
                        'labels': labels,
                        'label_idxs': label_idxs,
                        'lang': lang
                    }
                    numberized_data.append(inst)

            progress.close()
            with open(os.path.join(project_full_path, 'unlabeled-data.json'), 'w') as f:
                for nd in numberized_data:
                    f.write(json.dumps(nd) + '\n')
        return project_task_type


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def turn_doc_row_into_es_row(row, mapping_columns):
    new_row = dict((mapping_columns[col], row[col]) for col in row if col in mapping_columns)
    return new_row


def create_supervised_project(config,
                              detect_lang_fn,
                              file_bytes,
                              project_name,
                              project_type,
                              upload_id,
                              file_type,
                              upload_folder):
    """
    Saves the file in the labelled folder, counts number of lines, saves project in database.

    #TODO: optimise so we don't need to iterate through file twice
    #TODO (once for saving, another time for counting lines)
    """
    project_full_path = upload_folder

    uploaded_file = UploadedSupervisedFile(project_type,
                                           file_bytes,
                                           file_type)

    # perform document checks
    uploaded_file.do_all_file_checks()

    # Save project details in db
    project_id = project_name

    project_task_type = uploaded_file.process_file(config, detect_lang_fn, project_name, project_full_path)

    return project_id, project_task_type


def delete_files_from_disk(project_full_path):
    if os.path.exists(project_full_path):
        shutil.rmtree(project_full_path)
