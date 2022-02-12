'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/scripts/start_app.py
'''
import argparse
import os, json
import webbrowser
from famie.config.config_reader import read_config
from pathlib import Path


def main(args):
    ######### passing arguments ########
    with open(os.path.join(Path(__file__).parent.parent, 'api/active_learning/passed_args.json'), 'w') as f:
        json.dump({
            'selection': args.selection,
            'proxy_embedding': args.proxy_embedding,
            'target_embedding': args.target_embedding
        }, f)
    ####################################

    config = read_config()
    flask_port = config["DEFAULT"]["FLASK_PORT"]

    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:9000/')

    from famie.api import create_app

    application = create_app()
    application.config.from_mapping(config.items("DEFAULT"))

    application.run(debug=False,
                    port=flask_port,
                    host='0.0.0.0')
