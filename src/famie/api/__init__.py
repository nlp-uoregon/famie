'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/__init__.py
'''
from flask import Flask


def create_app():
    app = Flask(__name__)
    return app
