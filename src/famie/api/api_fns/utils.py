'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/api_fns/utils.py
'''
import io
import os


def get_column_names(file):
    first_line = file.readline()
    try:
        column_names = first_line.strip("\n").split(',')
    except:
        raise Exception("Need to load a csv file")
    file.seek(0)
    return column_names


def get_decoded_stream(file_bytes):
    file = io.TextIOWrapper(file_bytes, encoding='utf-8')
    return file


def check_file_size(file):
    lines = [line.strip() for line in file.readlines() if line.strip()]
    if len(lines) == 0:
        raise Exception("File is empty")
