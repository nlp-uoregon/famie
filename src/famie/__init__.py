import os, json, shutil
from collections import namedtuple
from famie.api.active_learning.constants import *


class Project:
    def __init__(self, project):
        self.project_name = project['name']
        self.type_set = [{'id': l['id'], 'type': l['name']} for l in project['classes']]

    def __str__(self):
        res = 'Project name: {}\n'.format(self.project_name)
        res += 'Type set: {}'.format(self.type_set)
        return res

    def get_labeled_data(self):
        print('Loading labeled data ...')
        print('-' * 50)
        labeled_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'labeled-data.json')
        if not os.path.exists(labeled_fpath):
            print('{} does not exist!'.format(labeled_fpath))
            return None
        else:
            with open(labeled_fpath) as f:
                data = json.load(f)
            print('Done!')
            return data

    def export_labeled_data(self, data_file):
        print('Exporting labeled data to `{}` ...'.format(data_file))
        print('-' * 50)

        labeled_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'labeled-data.json')
        if not os.path.exists(labeled_fpath):
            print('{} does not exist!'.format(labeled_fpath))
            return None
        else:
            with open(labeled_fpath) as f:
                data = json.load(f)

            with open(data_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False)
            print('Done!')

    def export_trained_model(self, model_file):
        print('Exporting trained model to `{}` ...'.format(model_file))
        print('-' * 50)

        saved_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'target_output_weights.ckpt')
        if not os.path.exists(saved_fpath):
            print('{} does not exist!'.format(saved_fpath))
            return None

        shutil.copy(saved_fpath, model_file)

        print('Done!')

    def get_trained_model(self):
        print('Loading trained model ...')
        print('-' * 50)
        from famie.api.active_learning.models import SeqLabel, XLMRobertaTokenizer, OUTPUT_DIR, WORKING_DIR

        saved_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'target_output_weights.ckpt')
        if not os.path.exists(saved_fpath):
            print('{} does not exist!'.format(saved_fpath))
            return None

        import torch
        import trankit

        torch.cuda.empty_cache()
        
        saved_ckpt = convert_json_to_ckpt(saved_fpath, use_gpu=torch.cuda.is_available())

        config = namedtuple('Config', field_names=[
            'cache_dir',
            'target_embedding_name',
            'proxy_embedding_name',
            'proxy_reduction_factor',
            'target_reduction_factor',
            'embedding_dropout',
            'vocabs',
            'hidden_num'
        ])

        config.cache_dir = os.path.join(WORKING_DIR, 'resource')
        config.target_embedding_name = saved_ckpt['embedding_name']
        config.proxy_embedding_name = saved_ckpt['embedding_name']
        config.proxy_reduction_factor = 2
        config.target_reduction_factor = 4
        config.embedding_dropout = 0.4
        config.vocabs = {
            self.project_name: saved_ckpt['vocabs']
        }
        config.hidden_num = 200
        config.max_sent_length = 200
        config.use_gpu = True if torch.cuda.is_available() else False

        config.trankit_tokenizer = trankit.Pipeline(saved_ckpt['lang'],
                                                    cache_dir=os.path.join(WORKING_DIR, 'cache/trankit'))
        config.target_tokenizer = XLMRobertaTokenizer.from_pretrained(config.target_embedding_name,
                                                                      cache_dir=os.path.join(config.cache_dir,
                                                                                             config.target_embedding_name),
                                                                      do_lower_case=False)

        trained_model = SeqLabel(
            config=config,
            project_id=self.project_name,
            model_name='target'
        )
        if config.use_gpu:
            trained_model.cuda()
            trained_model.half()

        trained_model.eval()
        trained_weights = saved_ckpt['weights']
        for name, param in trained_model.state_dict().items():
            if name not in trained_weights:
                trained_weights[name] = param

        trained_model.load_state_dict(trained_weights)
        print('Done!')
        return trained_model


def get_project(project_name):
    print('Loading project `{}` ...'.format(project_name))
    print('-' * 50)
    with open(PROJECT_INFO_FPATH) as f:
        project2info = json.load(f)

    if project_name in project2info:
        project_info = project2info[project_name]
        project = Project(project_info)
        print('Done!')
        return project
    else:
        print('Project `{}` does not exist!'.format(project_name))
        return None


def load_model_from_file(model_file):
    print('Loading trained model from `{}` ...'.format(model_file))
    print('-' * 50)
    if not os.path.exists(model_file):
        print('{} does not exist!'.format(model_file))
        return None

    from famie.api.active_learning.models import SeqLabel, XLMRobertaTokenizer, OUTPUT_DIR, WORKING_DIR

    import torch
    import trankit

    torch.cuda.empty_cache()

    config = namedtuple('Config', field_names=[
        'cache_dir',
        'target_embedding_name',
        'proxy_embedding_name',
        'proxy_reduction_factor',
        'target_reduction_factor',
        'embedding_dropout',
        'vocabs',
        'hidden_num'
    ])
    config.use_gpu = True if torch.cuda.is_available() else False

    saved_ckpt = convert_json_to_ckpt(model_file, use_gpu=config.use_gpu)

    config.cache_dir = os.path.join(WORKING_DIR, 'resource')
    config.target_embedding_name = saved_ckpt['embedding_name']
    config.proxy_embedding_name = saved_ckpt['embedding_name']
    config.proxy_reduction_factor = 2
    config.target_reduction_factor = 4
    config.embedding_dropout = 0.4
    config.vocabs = {
        saved_ckpt['project_name']: saved_ckpt['vocabs']
    }
    config.hidden_num = 200
    config.max_sent_length = 200

    config.trankit_tokenizer = trankit.Pipeline(saved_ckpt['lang'],
                                                cache_dir=os.path.join(WORKING_DIR, 'cache/trankit'))
    config.target_tokenizer = XLMRobertaTokenizer.from_pretrained(config.target_embedding_name,
                                                                  cache_dir=os.path.join(config.cache_dir,
                                                                                         config.target_embedding_name),
                                                                  do_lower_case=False)

    trained_model = SeqLabel(
        config=config,
        project_id=saved_ckpt['project_name'],
        model_name='target'
    )
    if config.use_gpu:
        trained_model.cuda()
        trained_model.half()

    trained_model.eval()
    trained_weights = saved_ckpt['weights']
    for name, param in trained_model.state_dict().items():
        if name not in trained_weights:
            trained_weights[name] = param

    trained_model.load_state_dict(trained_weights)
    print('Done!')
    return trained_model
