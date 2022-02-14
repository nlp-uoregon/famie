import os, json


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


CODE2LANG = {
    'af': 'afrikaans', 'ar': 'arabic', 'hy': 'armenian', 'eu': 'basque', 'be': 'belarusian', 'bg': 'bulgarian',
    'ca': 'catalan', 'zh': 'chinese', 'hr': 'croatian', 'cs': 'czech',
    'da': 'danish', 'nl': 'dutch', 'en': 'english', 'et': 'estonian', 'fi': 'finnish', 'fr': 'french', 'gl': 'galician',
    'de': 'german', 'el': 'greek', 'he': 'hebrew', 'hi': 'hindi',
    'hu': 'hungarian', 'id': 'indonesian', 'ga': 'irish', 'it': 'italian', 'ja': 'japanese', 'kk': 'kazakh',
    'ko': 'korean', 'ku': 'kurmanji', 'la': 'latin', 'lv': 'latvian',
    'lt': 'lithuanian', 'mr': 'marathi', 'nn': 'norwegian-nynorsk', 'nb': 'norwegian-bokmaal', 'fa': 'persian',
    'pl': 'polish', 'pt': 'portuguese', 'ro': 'romanian',
    'ru': 'russian', 'sr': 'serbian', 'sk': 'slovak', 'sl': 'slovenian', 'es': 'spanish',
    'sv': 'swedish', 'ta': 'tamil', 'te': 'telugu', 'tr': 'turkish',
    'uk': 'ukrainian', 'ur': 'urdu', 'ug': 'uyghur', 'vi': 'vietnamese'
}

SUPPORTED_TASKS = {
    'ner'
}

DEBUG = True

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
DATABASE_DIR = os.path.join(WORKING_DIR, 'database')
PROJECT_INFO_FPATH = os.path.join(DATABASE_DIR, 'project2info.json')
LOG_DIR = os.path.join(WORKING_DIR, 'logs')

OUTPUT_DIR = os.path.join(WORKING_DIR, 'famie-output')

SIGNAL_DIR = {'base': os.path.join(WORKING_DIR, 'signals')}
TASK_NAME_FPATH = os.path.join(SIGNAL_DIR['base'], 'task_name.txt')

ensure_dir(SIGNAL_DIR['base'])
ensure_dir(DATABASE_DIR)

for task in SUPPORTED_TASKS:
    SIGNAL_DIR[task] = os.path.join(SIGNAL_DIR['base'], task)
    ensure_dir(SIGNAL_DIR[task])

LISTEN_TIME = 1
MAX_EXAMPLES_PER_ITER = 10000

STOP_CONTROLLER = 'stop-controller'

PAUSE_MODEL = 'pause-model'
RUN_TARGET = 'run-target'
RUN_PROXY = 'run-proxy'

SIGNALS = {
    STOP_CONTROLLER: 'Stop the controller',
    PAUSE_MODEL: 'Pause model',
    RUN_TARGET: 'Run the target model',
    RUN_PROXY: 'Run the proxy model'
}

EMBEDDING2DIM = {
    'xlm-roberta-large': 1024,
    'xlm-roberta-base': 768,
    'nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large': 384,
    'nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large': 384
}

if not os.path.exists(PROJECT_INFO_FPATH):
    with open(PROJECT_INFO_FPATH, 'w') as f:
        json.dump({}, f)

SPAN_KEYS = ["id", "start", "end", "text", "entity_id"]

PROXY_BATCH_FIELDS = [
    'example_ids', 'texts', 'tokens', 'piece_idxs', 'attention_masks', 'token_lens',
    'label_idxs', 'token_nums', 'distill_mask',
    'tch_lbl_dist', 'transitions'
]

TARGET_BATCH_FIELDS = [
    'example_ids', 'texts', 'tokens', 'piece_idxs', 'attention_masks', 'token_lens',
    'label_idxs', 'token_nums', 'distill_mask'
]

AL_BATCH_FIELDS = [
    'example_ids', 'tokens', 'piece_idxs', 'attention_masks', 'token_lens',
    'labels', 'label_idxs', 'token_nums'
]

CKPT_KEYS = {'project_name', 'lang', 'embedding_name', 'hidden_num', 'vocabs', 'weights'}


def convert_ckpt_to_json(ckpt_fpath):
    import json, torch

    ckpt = torch.load(ckpt_fpath) if torch.cuda.is_available() else torch.load(ckpt_fpath, map_location=torch.device('cpu'))
    assert set(ckpt.keys()) == CKPT_KEYS

    for param_name in ckpt['weights']:
        ckpt['weights'][param_name] = ckpt['weights'][param_name].data.cpu().numpy().tolist()

    return ckpt


def convert_json_to_ckpt(json_fpath, use_gpu):
    import torch

    with open(json_fpath) as f:
        ckpt = json.load(f)

    assert set(ckpt.keys()) == CKPT_KEYS

    for param_name in ckpt['weights']:
        ckpt['weights'][param_name] = torch.tensor(ckpt['weights'][param_name]).cuda() if use_gpu else torch.tensor(
            ckpt['weights'][param_name])

    return ckpt
