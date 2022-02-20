'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/scripts/start_app.py
'''
import argparse
import os, json
import webbrowser
from famie.config.config_reader import read_config
from famie.api.active_learning.constants import WORKING_DIR
from pathlib import Path


def main(args):
    ######### passing arguments ########
    with open(os.path.join(WORKING_DIR, 'passed_args.json'), 'w') as f:
        json.dump({
            'selection': args.selection,
            'proxy_embedding': args.proxy_embedding,
            'target_embedding': args.target_embedding
        }, f)
    ####################################

    config = read_config()

    from famie.api import create_app

    application = create_app()

    from famie.api.blueprints.common import bp
    from famie.api.blueprints.supervised import supervised_bp

    application.register_blueprint(bp)
    application.register_blueprint(supervised_bp)

    application.config.from_mapping(config.items("DEFAULT"))

    print('FAMIE`s Web Interface is available at: http://127.0.0.1:{}/'.format(args.port))
    print('-' * 50)

    application.run(debug=False,
                    port=args.port,
                    host='0.0.0.0')
