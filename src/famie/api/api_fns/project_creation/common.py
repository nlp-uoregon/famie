'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/api_fns/project_creation/common.py
'''

from abc import ABC, abstractmethod
import csv
import json
from pathlib import Path
import random

from famie.api.api_fns.utils import check_file_size, get_column_names, get_decoded_stream
from famie.constants import (ES_GROUND_TRUTH_NAME_FIELD,
                             FILE_TYPE_DOCUMENTS,
                             FILE_TYPE_DOCUMENTS_WIKI,
                             INPUT_FILE_SPECS,
                             MAPPINGS,
                             PROJECT_TYPE_CLASSIFICATION,
                             PROJECT_TYPE_NER)


class UploadedFile(ABC):

    def __init__(self, project_type, input_data, file_type):
        # run input param checks
        if project_type in [PROJECT_TYPE_CLASSIFICATION, PROJECT_TYPE_NER]:
            if file_type not in [FILE_TYPE_DOCUMENTS, FILE_TYPE_DOCUMENTS_WIKI]:
                raise Exception(f"File type {file_type} is not supported for project type {project_type}.")

        if file_type == FILE_TYPE_DOCUMENTS_WIKI and project_type != PROJECT_TYPE_NER:
            raise Exception(f"Using urls is not supported for project type {project_type}.")

        self.project_type = project_type
        self.input_data = input_data
        if (type(input_data) != list):
            self.filename = input_data.filename
        self.total_documents = 0
        self.file_type = file_type  # documents, kb or documents_wiki
        self.processed_data = None

    def do_all_file_checks(self):
        self.input_data = get_decoded_stream(self.input_data)

    def read_line(self):
        num_lines = 0
        for line in self.input_data.readlines():
            line = line.strip()
            if line:
                yield line
                num_lines += 1

    def __iter__(self):
        for line in self.read_line():
            self.total_documents += 1
            yield {'text': line}

    @abstractmethod
    def process_file(self, es_uri, index_name, get_row, project_full_path, spacy_binary_filepath):
        pass


class ES_indexer(object):

    def __init__(self, es_uri, index_name, get_row, mapping_specs):
        self.es_uri = es_uri
        self.index_name = index_name
        self.mapping_es = mapping_specs["mapping_es"]
        self.settings_es = mapping_specs.get("settings_es")
        self.get_row = get_row
        self.bulk_line_size = 100
        self.num_read_lines = 0
        self.num_indexed_docs = 0
        self.current_rows = []

    def create_new_index(self):
        create_new_index(self.es_uri, self.index_name, self.mapping_es, self.settings_es)

    def __enter__(self):
        return self

    def index_line(self, line):
        if (self.num_read_lines > 0) and (self.num_read_lines % self.bulk_line_size == 0):
            try:
                bulk_load_documents(self.es_uri, self.index_name, self.current_rows, self.num_indexed_docs)
                self.num_indexed_docs = self.num_read_lines
                self.current_rows = []
            except Exception as e:
                print(f"Error bulk loading lines "
                      f"{self.num_indexed_docs} to {self.num_indexed_docs + len(self.current_rows) - 1} to elasticsearch")
                raise

        new_row = self.get_row(line)
        self.current_rows.append(new_row)
        self.num_read_lines += 1

    def __exit__(self, type, value, traceback):
        # do things at exit time
        if self.current_rows:
            try:
                bulk_load_documents(self.es_uri, self.index_name, self.current_rows, self.num_indexed_docs)
            except:
                print(f"Error bulk loading lines "
                      f"{self.num_indexed_docs} to {self.num_indexed_docs + len(self.current_rows) - 1} to elasticsearch")
                raise


def bulk_load_documents(es_uri, index_name, list_docs, start_doc_ind):
    json_data = []
    num_docs = 0
    for doc in list_docs:
        json_data.append(json.dumps({"index": {"_index": index_name,
                                               "_id": start_doc_ind + num_docs}}))
        if "id" not in doc:
            doc["id"] = start_doc_ind + num_docs
        json_data.append(json.dumps(doc))
        num_docs += 1
    request_body = "\n".join(json_data) + "\n"
    bulk_upload(es_uri, request_body)


def index_df(es_uri, index_name, df, get_row):
    num_lines = 100
    rows = []
    start_doc_ind = 0

    for ind, row in df.iterrows():
        if (ind > 0) and (ind % num_lines == 0):
            try:
                bulk_load_documents(es_uri, index_name, rows, start_doc_ind)
                start_doc_ind = ind
                rows = []
            except:
                print(f"Error bulk loading lines "
                      f"{start_doc_ind} to {start_doc_ind + num_lines - 1} to elasticsearch")
                raise

        new_row = get_row(row)
        rows.append(new_row)

    if rows:
        bulk_load_documents(es_uri, index_name, rows, start_doc_ind)


def get_random_int_5_digits():
    return random.randint(10000, 99999)


def sanitise_string(s):
    return ''.join(e for e in s if (e.isalnum()) or e == '_').lower()


def get_random_index_name(prefix):
    index_name = sanitise_string(prefix)
    suffix = get_random_int_5_digits()
    index_name = f"{index_name}_{suffix}"
    return index_name


def get_upload_key(project_type, file_type):
    if file_type not in INPUT_FILE_SPECS[project_type]:
        raise Exception(f"File type {file_type} not supported for project of type {project_type}.")
    return INPUT_FILE_SPECS[project_type][file_type]['upload_key']


def check_column_names(file, column_names):
    actual_column_names = get_column_names(file)
    for column_name in column_names:
        if not column_name in actual_column_names:
            raise Exception(f"File needs to contain a \"{column_name}\" column")
    return actual_column_names
