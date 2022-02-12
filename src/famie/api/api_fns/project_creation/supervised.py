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

    def process_file(self, project_name, project_full_path):
        if not os.path.exists(project_full_path):
            os.makedirs(project_full_path)

        json_writer = open(os.path.join(project_full_path, 'unlabeled-data.json'), 'w')

        lid = 0
        for line in self:
            json_writer.write(json.dumps({
                'project_name': project_name,
                'example_id': lid,
                'text': line['text']
            }) + '\n')
            lid += 1

        json_writer.close()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def turn_doc_row_into_es_row(row, mapping_columns):
    new_row = dict((mapping_columns[col], row[col]) for col in row if col in mapping_columns)
    return new_row


def create_supervised_project(file_bytes,
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

    uploaded_file.process_file(project_name, project_full_path)

    return project_id


def delete_files_from_disk(project_full_path):
    if os.path.exists(project_full_path):
        shutil.rmtree(project_full_path)
