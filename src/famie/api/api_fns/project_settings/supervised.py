'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/api_fns/project_settings/supervised.py
'''
from itertools import cycle
from famie.api.api_fns.utils import check_file_size, get_decoded_stream
from famie.constants import COLOURS


def add_class_colours(class_names):
    for class_colour, class_item in zip(cycle(COLOURS), class_names):
        class_item['colour'] = class_colour


def get_class_names(file_bytes):
    file = get_decoded_stream(file_bytes)
    lines = [line.strip() for line in file.readlines() if line.strip()]
    class_names = []
    for line_ind, line in enumerate(lines):
        class_name = line
        if len(class_name) == 0:
            raise Exception(f"There is an empty class name on line {line_ind + 1}.")
        class_names.append({"id": line_ind, "name": class_name})

    add_class_colours(class_names)
    return class_names


def set_class_names(file_bytes):
    class_names = get_class_names(file_bytes)
    return class_names


def convert_to_bio2(ori_tags):
    bio2_tags = []
    for i, tag in enumerate(ori_tags):
        if tag == 'O':
            bio2_tags.append(tag)
        elif tag[0] == 'I':
            if i == 0 or ori_tags[i - 1] == 'O' or ori_tags[i - 1][1:] != tag[1:]:
                bio2_tags.append('B' + tag[1:])
            else:
                bio2_tags.append(tag)
        else:
            bio2_tags.append(tag)
    return bio2_tags


def get_example_from_lines(sent_lines):
    tokens = []
    ner_tags = []
    for line in sent_lines:
        array = line.split()
        assert len(array) >= 2
        tokens.append(array[0])
        ner_tags.append(array[1])
    ner_tags = convert_to_bio2(ner_tags)
    return {'tokens': [{'text': t } for t in tokens], 'labels': ner_tags}


def get_examples_from_bio_fpath(raw_lines):
    sent_lines = []
    bio2_examples = []
    for line in raw_lines:
        line = line.strip()
        if '-DOCSTART-' in line:
            continue
        if len(line) > 0:
            array = line.split()
            if len(array) < 2:
                continue
            else:
                sent_lines.append(line)
        elif len(sent_lines) > 0:
            example = get_example_from_lines(sent_lines)
            bio2_examples.append(example)
            bio2_examples[-1]['example_id'] = 'provided-example-{}'.format(len(bio2_examples))

            sent_lines = []

    if len(sent_lines) > 0:
        bio2_examples.append(get_example_from_lines(sent_lines))
        bio2_examples[-1]['example_id'] = 'provided-example-{}'.format(len(bio2_examples))

    return bio2_examples


def parse_labeled_data(file_bytes):
    file = get_decoded_stream(file_bytes)
    lines = [line.strip() for line in file.readlines()]
    provided_labeled_data = get_examples_from_bio_fpath(lines)
    return provided_labeled_data
