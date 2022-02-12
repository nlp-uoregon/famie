import os, json, shutil
from collections import namedtuple


class Project:
    def __init__(self, project_name):
        self.project_name = project_name

    def get_labeled_data(self):
        from famie.api.active_learning.utils import OUTPUT_DIR
        labeled_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'labeled-data.json')
        if not os.path.exists(labeled_fpath):
            print('{} does not exist!'.format(labeled_fpath))
            return None
        else:
            with open(labeled_fpath) as f:
                data = json.load(f)
            return data

    def export_labeled_data(self, output_fpath):
        from famie.api.active_learning.utils import OUTPUT_DIR
        labeled_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'labeled-data.json')
        if not os.path.exists(labeled_fpath):
            print('{} does not exist!'.format(labeled_fpath))
            return None
        else:
            with open(labeled_fpath) as f:
                data = json.load(f)

            with open(output_fpath, 'w') as f:
                json.dump(data, f, ensure_ascii=False)


    def export_trained_model(self, output_fpath):
        from famie.api.active_learning.models import SeqLabel, XLMRobertaTokenizer, OUTPUT_DIR

        import torch
        import trankit

        saved_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'target_output_weights.ckpt')
        if not os.path.exists(saved_fpath):
            print('{} does not exist!'.format(saved_fpath))
            return None

        shutil.copy(saved_fpath, output_fpath)

    def get_trained_model(self):
        from famie.api.active_learning.models import SeqLabel, XLMRobertaTokenizer, OUTPUT_DIR, WORKING_DIR

        import torch
        import trankit

        saved_fpath = os.path.join(OUTPUT_DIR, self.project_name, 'target_output_weights.ckpt')
        if not os.path.exists(saved_fpath):
            print('{} does not exist!'.format(saved_fpath))
            return None
        saved_ckpt = torch.load(saved_fpath)

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

        config.trankit_tokenizer = trankit.Pipeline(saved_ckpt['lang'], cache_dir=os.path.join(WORKING_DIR, 'cache/trankit'))
        config.target_tokenizer = XLMRobertaTokenizer.from_pretrained(config.target_embedding_name,
                                                                      cache_dir=os.path.join(config.cache_dir,
                                                                                             config.target_embedding_name),
                                                                      do_lower_case=False)

        trained_model = SeqLabel(
            config=config,
            project_id=self.project_name,
            model_name='target'
        )
        trained_model.cuda()
        trained_model.eval()
        trained_weights = saved_ckpt['weights']
        for name, param in trained_model.state_dict().items():
            if name not in trained_weights:
                trained_weights[name] = param

        trained_model.load_state_dict(trained_weights)

        return trained_model


def get_project(project_name):
    return Project(project_name)
