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
