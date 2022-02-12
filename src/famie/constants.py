'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/constants.py
'''
from pathlib import Path

HOME = str(Path.home())

ROOT_PATH = str(Path(__file__).parent.absolute())

# types of projects
PROJECT_TYPE_CLASSIFICATION = "classification"
PROJECT_TYPE_NER = "ner"
PROJECT_TYPE_ED = "ed"

ALL_PROJECT_TYPES = [PROJECT_TYPE_CLASSIFICATION, PROJECT_TYPE_NER, PROJECT_TYPE_ED]

# input file columns
TEXT_COLUMN_NAME = 'text'
TABLE_COLUMN_NAMES_FIELD_NAME = 'column_names'
TABLE_ROWS_FIELD_NAME = 'rows'
TABLE_ROWS_CHAR_STARTS_FIELD_NAME = "char_starts"
GROUND_TRUTH_COLUMN_NAME = "__LABEL__"
MENTIONS_COLUMN_NAME = "mentions"

FILE_TYPE_DOCUMENTS = "documents"
FILE_TYPE_DOCUMENTS_WIKI = "documents_wiki"
FILE_TYPE_KB = "kb"

INPUT_FILE_SPECS = {
    PROJECT_TYPE_CLASSIFICATION: {FILE_TYPE_DOCUMENTS:
                                      {"required": [TEXT_COLUMN_NAME],
                                       "optional": [GROUND_TRUTH_COLUMN_NAME],
                                       "text": [TEXT_COLUMN_NAME],
                                       "upload_key": "upload_id",
                                       "max_rows": 3000}},
    PROJECT_TYPE_NER: {FILE_TYPE_DOCUMENTS: {"required": [],
                                             "text": [TEXT_COLUMN_NAME],
                                             "upload_key": "upload_id",
                                             "max_rows": 100000},
                       FILE_TYPE_DOCUMENTS_WIKI: {"required": ["url"],
                                                  "text": [TEXT_COLUMN_NAME],
                                                  "upload_key": "upload_id",
                                                  "max_rows": 400}},
    PROJECT_TYPE_ED: {FILE_TYPE_DOCUMENTS: {"required": [TEXT_COLUMN_NAME, MENTIONS_COLUMN_NAME],
                                            "text": [TEXT_COLUMN_NAME],
                                            "upload_key": "upload_id",
                                            "max_rows": 3000},
                      FILE_TYPE_KB: {"required": ["name", "description"],
                                     "text": ["description"],
                                     "upload_key": "kb_upload_id"}}
}

SPACY_COLUMN_NAME = "__spacy_doc__"

ES_TEXT_FIELD_NAME = "text"
ES_GROUND_TRUTH_NAME_FIELD = "__LABEL__"
ES_GROUND_TRUTH_LABEL_FIELD = "__LABEL_id__"

MAPPING_CLASSIFICATION = {
    "properties": {
        ES_TEXT_FIELD_NAME: {"type": "text"},
        "rules": {"type": "nested",
                  "properties": {
                      "rule_id": {"type": "integer"},
                      "label": {"type": "integer"}}},
        "predicted_label": {"type": "integer"},
        "manual_label": {"properties":
                             {"label": {"type": "integer"},
                              "session": {"type": "keyword"}}},
        "id": {"type": "integer"},
        ES_GROUND_TRUTH_NAME_FIELD: {"type": "keyword"},
        ES_GROUND_TRUTH_LABEL_FIELD: {"type": "integer"}
    }
}

MAPPING_NER = {
    "properties": {
        ES_TEXT_FIELD_NAME: {"type": "text"},
        "url": {"type": "keyword"},  # Does the data come from a url (wiki)?
        "is_table": {"type": "boolean"},  # Is the data to be displayed as a table?
        TABLE_ROWS_FIELD_NAME: {"type": "text"},  # array of arrays
        TABLE_COLUMN_NAMES_FIELD_NAME: {"type": "text"},  # array of column names
        TABLE_ROWS_CHAR_STARTS_FIELD_NAME: {"type": "integer"}, # array of integers with char starts of each row
        "paragraph_id": {"type": "integer"},
        "rules": {"type": "nested",
                  "properties": {
                      "rule_id": {"type": "integer"},
                      "label": {"type": "nested",
                                "properties":
                                    {"id": {"type": "keyword"},
                                     "start": {"type": "integer"},
                                     "end": {"type": "integer"},
                                     "entity_id": {"type": "integer"},
                                     "text": {"type": "text"}}}}},
        "manual_label": {"properties": {
            "label": {"type": "nested",
                      "properties":
                          {"id": {"type": "keyword"},
                           "start": {"type": "integer"},
                           "end": {"type": "integer"},
                           "entity_id": {"type": "integer"},
                           "text": {"type": "text"}}},
            "session": {"type": "keyword"}}},
        "predicted_label": {"properties": {
            "id": {"type": "keyword"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "entity_id": {"type": "integer"},
            "text": {"type": "text"}}},
        "id": {"type": "integer"}
    }
}

MAPPING_MENTIONS = {
    "properties": {
        ES_TEXT_FIELD_NAME: {"type": "text"},
        "label": {"properties": {
            "id": {"type": "keyword"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "text": {"type": "text"}}},
        "id": {"type": "integer"}
    }
}

MAPPING_KB = {
    "properties": {
        ES_TEXT_FIELD_NAME: {"type": "text",
                             "analyzer": "stem_analyzer"},
        "name": {"type": "text",
                 "analyzer": "stem_analyzer",
                 "fields": {
                     "raw": {
                         "type": "keyword"
                     }
                 }},
        "id": {"type": "integer"},
        "colour": {"type": "keyword"}
    }
}

SETTINGS_KB = {
    "analysis": {
        "analyzer": {
            "stem_analyzer": {
                "tokenizer": "standard",  # tried whitespace and gave weird results
                "filter": [
                    "lowercase",
                    "porter_stem"
                ]
            }
        }
    }
}

ABSTAIN = -1

# mapping_columns are not the required columns, it's the name mapping
MAPPINGS = {PROJECT_TYPE_CLASSIFICATION:
                {FILE_TYPE_DOCUMENTS: {"mapping_es": MAPPING_CLASSIFICATION,
                                       "mapping_columns": {
                                           TEXT_COLUMN_NAME: ES_TEXT_FIELD_NAME,
                                           GROUND_TRUTH_COLUMN_NAME: ES_GROUND_TRUTH_NAME_FIELD}}
                 },
            PROJECT_TYPE_NER:
                {FILE_TYPE_DOCUMENTS: {"mapping_es": MAPPING_NER,
                                       "mapping_columns": {TEXT_COLUMN_NAME: ES_TEXT_FIELD_NAME}},
                 FILE_TYPE_DOCUMENTS_WIKI: {"mapping_es": MAPPING_NER,
                                            "mapping_columns": {TEXT_COLUMN_NAME: ES_TEXT_FIELD_NAME,
                                                                TABLE_ROWS_FIELD_NAME: TABLE_ROWS_FIELD_NAME,
                                                                TABLE_COLUMN_NAMES_FIELD_NAME:
                                                                    TABLE_COLUMN_NAMES_FIELD_NAME,
                                                                TABLE_ROWS_CHAR_STARTS_FIELD_NAME:
                                                                    TABLE_ROWS_CHAR_STARTS_FIELD_NAME,
                                                                "url": "url",
                                                                "is_table": "is_table",
                                                                "paragraph_id": "paragraph_id"}}},
            PROJECT_TYPE_ED:
                {FILE_TYPE_DOCUMENTS: {"mapping_es": MAPPING_MENTIONS,
                                       "mapping_columns": {TEXT_COLUMN_NAME: ES_TEXT_FIELD_NAME,
                                                           MENTIONS_COLUMN_NAME: "label"}},
                 FILE_TYPE_KB: {"mapping_es": MAPPING_KB,
                                "settings_es": SETTINGS_KB,
                                "mapping_columns": {"description": ES_TEXT_FIELD_NAME,
                                                    "name": "name"}}}}

SUPPORTED_RULES = {PROJECT_TYPE_CLASSIFICATION: ['ordered', 'non-ordered', 'sentiment'],
                   PROJECT_TYPE_NER: ['entity_regex', 'noun_phrase_regex']}

SPAN_KEYS = ["id", "start", "end", "text", "entity_id"]
NUM_MAX_SPAN_TOKENS = 6

# Request page sizes
ENTITY_PAGE_SIZE = 10  # number of entities returned
SINGLE_ENTITY_DOCS_PAGE_SIZE = 10  # number of documents to return when querying single entity
MULTIPLE_ENTITY_DOCS_PAGE_SIZE = 1  # number of documents to return when querying multiple entities

# Colours for the classes
COLOURS = ['deepOrange',
           'purple',
           'teal',
           'lightGreen',
           'cyan',
           'blueGrey',
           'blue',
           'orange',
           'yellow',
           'green',
           'grey',
           'pink',
           'amber',
           'indigo',
           'deepPurple',
           'red',
           'lightBlue',
           'brown',
           'lime']

# Modelling constants
NUM_ITERATIONS_BOOTSTRAP = 100
MIN_LABELLED_PRECISION = 5  # minimum number of predicted labels of a class that have been manually labelled
MIN_LABELLED_RECALL = 5  # minimum number of manual labels assigned to a specific class
