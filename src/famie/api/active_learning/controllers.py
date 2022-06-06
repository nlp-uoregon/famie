# define controllers for active learning modules
from .models import SeqLabel, ConditionalSeqLabel
from .trainers import *
from .utils import *

import time, os
import _thread as thread


class Controller:
    def __init__(self, config, task):
        assert task in SUPPORTED_TASKS

        self.config = config
        self.task = task
        self.signal_fpath = os.path.join(SIGNAL_DIR[task], 'signal.controller.txt')

        self.reset_signal()

        self.model = None
        self.dataset = None
        self.trainer = None
        self.is_listening = False

    def reset_signal(self):
        with open(self.signal_fpath, 'w') as f:
            f.write(PAUSE_MODEL)

    def read_signal(self):
        with open(self.signal_fpath) as f:
            signal = f.read().strip().lower()

        return signal

    def receive_signal(self, signal):
        with open(self.signal_fpath, 'w') as f:
            f.write(signal)

    def stop(self):
        self.receive_signal(STOP_CONTROLLER)

    def run_proxy_model(self, unlabeled_data, project_name):
        self.trainer['proxy'].receive_signal(RUN_PROXY, unlabeled_data, project_name)
        self.trainer['target'].receive_signal(RUN_PROXY, unlabeled_data, project_name)

    def run_target_model(self, project_name):
        self.trainer['proxy'].receive_signal(RUN_TARGET, [], project_name)
        self.trainer['target'].receive_signal(RUN_TARGET, [], project_name)

    def proxy_model_predicts(self, unlabeled_data, project_name):
        if self.trainer['proxy'].is_trained:
            self.trainer['proxy'].receive_signal(PROXY_PREDICTS, unlabeled_data, project_name)
        if self.trainer['target'].is_trained:
            self.trainer['target'].receive_signal(PROXY_PREDICTS, unlabeled_data, project_name)

    def target_model_predicts(self, unlabeled_data, project_name):
        if self.trainer['proxy'].is_trained:
            self.trainer['proxy'].receive_signal(TARGET_PREDICTS, unlabeled_data, project_name)
        if self.trainer['target'].is_trained:
            self.trainer['target'].receive_signal(TARGET_PREDICTS, unlabeled_data, project_name)

    def stop_listening(self):
        if self.trainer:
            self.trainer['proxy'].receive_signal(STOP_CONTROLLER, [], None)
            self.trainer['target'].receive_signal(STOP_CONTROLLER, [], None)

        self.model = None
        self.dataset = None
        self.trainer = None
        self.is_listening = False
        # print('{} controller: stopped listening!'.format(self.task))

    def listen(self, project_state):
        project_dir = project_state['project_dir']
        project_id = project_state['project_id']
        project_task_type = project_state['project_task_type']
        project_annotations = project_state['annotations']
        provided_labeled_data = project_state['provided_labeled_data']

        if not self.is_listening:
            print('-' * 50)
            print("Loading models for project '{}'...".format(project_id))
            self.is_listening = True

            self.config.vocabs[project_id] = {
                'entity-type': {}, 'entity-label': {'O': 0}
            }
            with open(os.path.join(project_dir, 'vocabs.json')) as f:
                self.config.vocabs[project_id]['entity-type'] = json.load(f)

            for entity_type in self.config.vocabs[project_id]['entity-type']:
                self.config.vocabs[project_id]['entity-label']['B-{}'.format(entity_type)] = len(
                    self.config.vocabs[project_id]['entity-label'])
                self.config.vocabs[project_id]['entity-label']['I-{}'.format(entity_type)] = len(
                    self.config.vocabs[project_id]['entity-label'])

            self.dataset = {
                'proxy': ProxyDataset(self.config, project_id, project_dir, project_annotations, provided_labeled_data),
                'target': TargetDataset(self.config, project_id, project_dir, project_annotations,
                                        provided_labeled_data)
            }
            self.dataset['target'].lang = self.dataset['proxy'].lang

            if project_task_type == 'conditional':
                print('initializating ConditionalSeqLabel models...')
                self.model = {
                    'proxy': ConditionalSeqLabel(self.config, project_id, model_name='proxy'),
                    'target': ConditionalSeqLabel(self.config, project_id, model_name='target')
                }
            else:
                print('initializing SeqLabel models...')
                assert project_task_type == 'unconditional'
                self.model = {
                    'proxy': SeqLabel(self.config, project_id, model_name='proxy'),
                    'target': SeqLabel(self.config, project_id, model_name='target')
                }

            self.trainer = {
                'proxy': ProxyTrainer(self.config, self.task, self.model['proxy'], self.dataset['proxy'],
                                      project_task_type),
                'target': TargetTrainer(self.config, self.task, self.model['target'], self.dataset['target'],
                                        project_task_type)
            }

            thread.start_new_thread(self.trainer['proxy'].start_listening, ())
            thread.start_new_thread(self.trainer['target'].start_listening, ())
            print('-' * 50)
        else:
            self.dataset['proxy'].update_data(project_annotations, provided_labeled_data)
            self.dataset['target'].update_data(project_annotations, provided_labeled_data)


if __name__ == '__main__':
    controller = Controller({}, 'ner')
    controller.listen()
    print('Listening ...')
    while True:
        time.sleep(1)
