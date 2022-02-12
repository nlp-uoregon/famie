'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/api/__init__.py
'''
from flask import Flask
from famie.api.blueprints.common import bp
from famie.api.blueprints.supervised import supervised_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(bp)
    app.register_blueprint(supervised_bp)

    return app
