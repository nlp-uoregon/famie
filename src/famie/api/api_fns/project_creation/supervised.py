'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/api_fns/project_creation/supervised.py
'''

import os
import json
import sys
import shutil
import traceback

from werkzeug.utils import secure_filename

from famie.api.api_fns.project_creation.common import (ES_indexer,
                                                       get_random_index_name,
                                                       UploadedFile)
from famie.constants import MAPPINGS

ALLOWED_EXTENSIONS = {'csv', 'txt'}


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
        numberized_data = []
        with open(os.path.join(project_full_path, 'unlabeled-data.raw.json')) as f:
            for line in f:
                line = line.strip()
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
                        'example_id': d['example_id'],
                        'text': d['text'],
                        'lang': lang,
                        'tokens': trankit_tokens,
                        'piece_idxs': piece_idxs,
                        'attention_mask': attn_masks,
                        'token_lens': s_token_lens,
                        'labels': labels,
                        'label_idxs': label_idxs
                    }
                    numberized_data.append(inst)

        with open(os.path.join(project_full_path, 'unlabeled-data.json'), 'w') as f:
            for nd in numberized_data:
                f.write(json.dumps(nd) + '\n')


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

    uploaded_file.process_file(config, detect_lang_fn, project_name, project_full_path)

    return project_id


def delete_files_from_disk(project_full_path):
    if os.path.exists(project_full_path):
        shutil.rmtree(project_full_path)
