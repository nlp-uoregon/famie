# define data processing/writing utility functions here
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb
import os, time, json, random
from copy import deepcopy
import langid
from transformers import XLMRobertaModel, AdapterConfig
from collections import Counter, namedtuple, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import langid
import tqdm
import trankit
from collections import Counter, namedtuple, defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from .constants import *

for code in CODE2LANG:
    assert CODE2LANG[code] in trankit.supported_langs

langid.set_languages([code for code in CODE2LANG])


def check_entity_form_input(spans):
    for span in spans:
        if len(span) == 0:
            continue
        if not all(k in span for k in SPAN_KEYS):
            raise Exception(f"Need to send all these keys in a span: {SPAN_KEYS}."
                            f"Current span: {span}")
    return spans


def update_project_to_database(project_info):
    with open(PROJECT_INFO_FPATH) as f:
        project2info = json.load(f)

    project2info[project_info['name']] = project_info

    with open(PROJECT_INFO_FPATH, 'w') as f:
        json.dump(project2info, f, ensure_ascii=False)


def get_project_info(project_name):
    with open(PROJECT_INFO_FPATH) as f:
        project2info = json.load(f)
    if project_name in project2info:
        return project2info[project_name]
    else:
        return None


def random_select(unlabeled_examples, selected_size=10):
    idxs = list(range(len(unlabeled_examples)))
    random.seed(len(unlabeled_examples))
    random.shuffle(idxs)
    selected = [unlabeled_examples[idx] for idx in idxs[:selected_size]]
    return selected


def get_examples(project_name):
    if os.path.exists(os.path.join(DATABASE_DIR, project_name, 'annotations.json')):
        with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json')) as f:
            id2annotations = {}
            for line in f:
                line = line.strip()
                if line:
                    annotations = json.loads(line)
                    # new annotations for the same example will override the old annotations
                    id2annotations[annotations['example_id']] = annotations['spans']
    else:
        id2annotations = {}

    unlabeled = []
    with open(os.path.join(DATABASE_DIR, project_name, 'unlabeled-data.json')) as f:
        examples = [json.loads(line.strip()) for line in f if line.strip()]

    for ex in examples:
        if ex['example_id'] not in id2annotations:
            inst = deepcopy(ex)
            inst['project_name'] = project_name
            inst['example_id'] = ex['example_id']
            inst['text'] = ex['text']
            inst['is_table'] = False
            inst['column_names'] = []
            inst['rows'] = []
            inst['char_starts'] = []
            inst['label'] = None
            inst['manual_label'] = None
            inst['rules'] = []

            unlabeled.append(inst)

    return id2annotations, unlabeled


def get_project_stats(project_name):
    with open(PROJECT_INFO_FPATH) as f:
        project2info = json.load(f)

    assert project_name in project2info
    info = project2info[project_name]

    fpath = os.path.join(DATABASE_DIR, project_name, 'unlabeled-data.json')
    with open(fpath) as f:
        data_size = len([line.strip() for line in f if line.strip()])
    info['stats'] = {
        'docs': {
            'total_rules_overlaps': None,
            'classes': [],
            'index_name': project_name,
            'is_wiki': False,
            'filename': os.path.join(DATABASE_DIR, project_name, 'unlabeled-data.json'),
            'name': project_name,
            'type': 'ner',
            'upload_id': 'unknown', 'supervised_type': 'ner',
            'total_manual_docs_empty': 0, 'total_documents': data_size,
            'total_manual_docs': 0
        },
        'rules': []
    }
    classes = info['classes']
    for l in classes:
        l['total_manual_docs'] = 0  # we will not update this because it is not very informative at label level
        l['total_manual_spans'] = 0

        info['stats']['docs']['classes'].append(l)

    if os.path.exists(os.path.join(DATABASE_DIR, project_name, 'annotations.json')):
        with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json')) as f:
            id2annotations = {}
            for line in f:
                line = line.strip()
                if line:
                    annotations = json.loads(line)
                    # new annotations for the same example will override the old annotations
                    id2annotations[annotations['example_id']] = annotations['spans']

        with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json'), 'w') as f:
            for exid in id2annotations:
                f.write(json.dumps({
                    'example_id': exid,
                    'spans': id2annotations[exid]
                }) + '\n')

        with open(os.path.join(DATABASE_DIR, project_name, 'annotations.json')) as f:
            for line in f:
                line = line.strip()
                if line:
                    info['stats']['docs']['total_manual_docs'] += 1
                    annotations = json.loads(line)
                    for span in annotations['spans']:
                        info['stats']['docs']['classes'][span['entity_id']]['total_manual_spans'] += 1

    return info['stats']


def get_project_list():
    with open(PROJECT_INFO_FPATH) as f:
        project2info = json.load(f)

    all_projects = []
    for project_name in project2info:
        project = project2info[project_name]

        all_projects.append({"project_id": project['id'],
                             "project_name": project['name'],
                             "project_type": project['type'],
                             "class_names": [{"id": class_name['id'],
                                              "name": class_name['name'],
                                              "colour": class_name['colour']}
                                             for class_name in project['classes']],
                             "rules": [],
                             "project_upload_finished": project['index_name'] is not None,
                             "project_params_finished": len(project['classes']) > 0,
                             "filenames": {"documents": project['filename']},
                             "wiki_data": False})
    return all_projects


def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def tag_paths_to_spans(paths, token_nums, vocab):
    """Convert predicted tag paths to a list of spans (entity mentions or event
    triggers).
    :param paths: predicted tag paths.
    :return (list): a list (batch) of lists (sequence) of spans.
    """
    batch_mentions = []
    itos = {i: s for s, i in vocab.items()}
    for i, path in enumerate(paths):
        mentions = []
        cur_mention = None
        path = path.tolist()[:token_nums[i].item()]
        for j, tag in enumerate(path):
            tag = itos[tag]
            if tag == 'O':
                prefix = tag = 'O'
            else:
                prefix, tag = tag.split('-', 1)

            if prefix == 'B':
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = [j, j + 1, tag]
            elif prefix == 'I':
                if cur_mention is None:
                    # treat it as B-*
                    cur_mention = [j, j + 1, tag]
                elif cur_mention[-1] == tag:
                    cur_mention[1] = j + 1
                else:
                    # treat it as B-*
                    mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
            else:
                if cur_mention:
                    mentions.append(cur_mention)
                cur_mention = None
        if cur_mention:
            mentions.append(cur_mention)
        mentions = [(i, j, tag) for i, j, tag in mentions]
        batch_mentions.append(mentions)
    return batch_mentions


def detect_lang(text):
    detected_code = langid.classify(text)[0]
    detected_lang = CODE2LANG[detected_code]
    print('-' * 50)
    print('Detected Language: {}'.format(detected_lang))
    print('-' * 50)
    return detected_lang


def create_charmap(tokens):
    char2tok = {}
    for tid, tok in enumerate(tokens):
        for cid in range(tok['span'][0], tok['span'][1]):
            char2tok[cid] = tid
    return char2tok


def convert_to_toklevel(charlevel_spans, charmap, itos_vocab):
    toklevel_spans = []
    for span in charlevel_spans:
        cstart = span['start']
        tstart = -1
        while tstart == -1 and cstart <= span['end']:
            if cstart in charmap:
                tstart = charmap[cstart]
            cstart += 1

        cend = span['end']
        tend = -1
        while tend == -1 and cend >= span['start']:
            if cend in charmap:
                tend = charmap[cend]
            cend -= 1

        if tstart >= 0 and tend >= 0:
            toklevel_spans.append({
                'start': tstart, 'end': tend + 1,
                'type': itos_vocab[span['entity_id']]
            })
    return toklevel_spans


ProxyBatch = namedtuple('ProxyBatch', field_names=PROXY_BATCH_FIELDS)

TargetBatch = namedtuple('TargetBatch', field_names=TARGET_BATCH_FIELDS)


def subword_tokenize(tokens, tokenizer, max_sent_length):
    s_pieces = [[p for p in tokenizer.tokenize(w['text']) if p != '▁'] for w in tokens]
    for ps in s_pieces:
        if len(ps) == 0:
            ps += ['-']
    s_token_lens = [len(ps) for ps in s_pieces]
    s_pieces = [p for ps in s_pieces for p in ps]
    # Pad word pieces with special tokens
    piece_idxs = tokenizer.encode(
        s_pieces,
        add_special_tokens=True,
        max_length=max_sent_length,
        truncation=True
    )
    if len(piece_idxs) > max_sent_length or len(piece_idxs) == 0:
        return None, None, None

    pad_num = max_sent_length - len(piece_idxs)
    attn_masks = [1] * len(piece_idxs) + [0] * pad_num
    piece_idxs = piece_idxs + [0] * pad_num
    return piece_idxs, attn_masks, s_token_lens


class ProxyDataset(Dataset):
    def __init__(self, config, project_id, project_dir, project_annotations):
        self.config = config

        self.project_id = project_id
        self.project_dir = project_dir
        self.distil_file = os.path.join(project_dir, 'distillations.json')
        self.project_annotations = project_annotations

        unlabeled_sample = None
        with open(os.path.join(self.project_dir, 'unlabeled-data.json')) as f:
            for line in f:
                line = line.strip()
                if line:
                    unlabeled_sample = json.loads(line)
                    break
        if unlabeled_sample:
            self.lang = unlabeled_sample['lang']
        else:
            self.lang = 'english'

        self.trankit_dir = os.path.join(project_dir, 'trankit_features')
        ensure_dir(self.trankit_dir)
        self.data = []
        self.numberize()
        self.batch_num = len(self.data) // config.batch_size

    def update_data(self, project_annotations):
        self.project_annotations = project_annotations
        self.numberize()
        self.batch_num = len(self.data) // self.config.batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        if self.lang != self.config.trankit_tokenizer.active_lang:
            if self.lang not in self.config.trankit_tokenizer.added_langs:
                self.config.trankit_tokenizer.add(self.lang)
            self.config.trankit_tokenizer.set_active(self.lang)
            print('Using {} tokenizer...'.format(self.lang))

        id2tch_lbl_dist = {}
        id2transitions = {}

        if self.config.distill and os.path.exists(self.distil_file):
            with open(self.distil_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        content = json.loads(line)
                        id2tch_lbl_dist[content['example_id']] = content['signals']['tch_lbl_dist']
                        id2transitions[content['example_id']] = content['signals']['transitions']

            # print('Loaded distillation signals for {} examples'.format(len(id2tch_lbl_dist)))

        itos_vocab = {i: s for s, i in self.config.vocabs[self.project_id]['entity-type'].items()}

        self.data = []
        with open(os.path.join(self.project_dir, 'unlabeled-data.json')) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d['example_id'] not in self.project_annotations:
                        continue

                    charlevel_spans = self.project_annotations[d['example_id']]
                    if not os.path.exists(os.path.join(self.trankit_dir, '{}.json'.format(d['example_id']))):
                        tokens = d['tokens']
                        ######### subword tokenization #######
                        proxy_piece_idxs, proxy_attn_masks, proxy_token_lens = subword_tokenize(
                            tokens, self.config.proxy_tokenizer,
                            self.config.max_sent_length
                        )
                        target_piece_idxs, target_attn_masks, target_token_lens = subword_tokenize(
                            tokens, self.config.target_tokenizer,
                            self.config.max_sent_length
                        )
                        ######### read annotations ###########
                        charmap = create_charmap(tokens)
                        toklevel_spans = convert_to_toklevel(charlevel_spans, charmap, itos_vocab)

                        labels = ['O'] * len(tokens)
                        for span in toklevel_spans:
                            labels[span['start']] = 'B-{}'.format(span['type'])
                            for k in range(span['start'] + 1, span['end']):
                                labels[k] = 'I-{}'.format(span['type'])

                        label_idxs = [self.config.vocabs[self.project_id]['entity-label'].get(label, 0) for label in
                                      labels]
                        inst = {
                            'example_id': d['example_id'],
                            'text': d['text'],
                            'tokens': tokens,

                            'proxy_piece_idxs': proxy_piece_idxs,
                            'proxy_attention_mask': proxy_attn_masks,
                            'proxy_token_lens': proxy_token_lens,

                            'target_piece_idxs': target_piece_idxs,
                            'target_attention_mask': target_attn_masks,
                            'target_token_lens': target_token_lens,

                            'labels': labels,
                            'label_idxs': label_idxs,
                            # 'charlevel_spans': charlevel_spans,
                            'toklevel_spans': toklevel_spans
                        }
                        with open(os.path.join(self.trankit_dir, '{}.json'.format(d['example_id'])), 'w') as f:
                            json.dump(inst, f, ensure_ascii=False)
                    else:
                        with open(os.path.join(self.trankit_dir, '{}.json'.format(d['example_id']))) as f:
                            inst = json.load(f)
                    if d['example_id'] in id2tch_lbl_dist and d['example_id'] in id2transitions:
                        inst['tch_lbl_dist'] = id2tch_lbl_dist[d['example_id']]
                        inst['transitions'] = id2transitions[d['example_id']]
                    else:
                        inst['tch_lbl_dist'] = []
                        inst['transitions'] = [0] * (len(inst['tokens']) + 1)

                    self.data.append(inst)

        torch.cuda.empty_cache()

        self.output_labeled_data()

    def output_labeled_data(self):
        ensure_dir(os.path.join(OUTPUT_DIR, self.project_id))
        output_data = [{
            'example_id': inst['example_id'],
            'text': inst['text'],
            'tokens': [t['text'] for t in inst['tokens']],
            'labels': inst['labels']
        } for inst in self.data]
        with open(os.path.join(OUTPUT_DIR, self.project_id, 'labeled-data.json'), 'w') as f:
            json.dump(output_data, f)

    def collate_fn(self, batch):
        example_ids = [inst['example_id'] for inst in batch]
        batch_tokens = [inst['tokens'] for inst in batch]
        batch_token_nums = [len(inst['tokens']) for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_lens = []

        max_token_num = max(batch_token_nums)

        batch_labels = []

        batch_tch_lbl_dist = []

        batch_transitions = []

        for inst in batch:
            token_num = len(inst['tokens'])
            batch_piece_idxs.append(inst['proxy_piece_idxs'])
            batch_attention_masks.append(inst['proxy_attention_mask'])
            batch_token_lens.append(inst['proxy_token_lens'])
            # for identification
            batch_labels.append(inst['label_idxs'] +
                                [0] * (max_token_num - token_num))

            # for distillation
            batch_tch_lbl_dist.append(
                inst['tch_lbl_dist'] + [[0] * len(self.config.vocabs[self.project_id]['entity-label'])] * (
                        max_token_num - len(inst['tch_lbl_dist']))
            )

            # transition scores
            batch_transitions.append(
                inst['transitions'] + [0] * (max_token_num - token_num)
            )

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs) if self.config.use_gpu else torch.LongTensor(
            batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks) if self.config.use_gpu else torch.FloatTensor(
            batch_attention_masks)

        batch_labels = torch.cuda.LongTensor(batch_labels) if self.config.use_gpu else torch.LongTensor(batch_labels)
        batch_token_nums = torch.cuda.LongTensor(batch_token_nums) if self.config.use_gpu else torch.LongTensor(
            batch_token_nums)

        batch_tch_lbl_dist = torch.cuda.FloatTensor(batch_tch_lbl_dist) if self.config.use_gpu else torch.FloatTensor(
            batch_tch_lbl_dist)
        distill_mask = torch.cuda.FloatTensor(
            [len(inst['tch_lbl_dist']) > 0 for inst in batch]) if self.config.use_gpu else torch.FloatTensor(
            [len(inst['tch_lbl_dist']) > 0 for inst in batch])

        batch_transitions = torch.cuda.FloatTensor(batch_transitions) if self.config.use_gpu else torch.FloatTensor(
            batch_transitions)

        return ProxyBatch(
            example_ids=example_ids,
            texts=[inst['text'] for inst in batch],
            tokens=batch_tokens,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_lens=batch_token_lens,
            label_idxs=batch_labels,
            token_nums=batch_token_nums,
            tch_lbl_dist=batch_tch_lbl_dist,
            transitions=batch_transitions,
            distill_mask=distill_mask
        )


class TargetDataset(Dataset):
    def __init__(self, config, project_id, project_dir, project_annotations):
        self.config = config

        self.project_id = project_id
        self.project_dir = project_dir
        self.project_annotations = project_annotations
        self.trankit_dir = os.path.join(project_dir, 'trankit_features')
        ensure_dir(self.trankit_dir)
        self.data = []
        self.numberize()
        self.batch_num = len(self.data) // config.batch_size

    def update_data(self, project_annotations):
        self.project_annotations = project_annotations
        self.numberize()
        self.batch_num = len(self.data) // self.config.batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        self.data = []
        with open(os.path.join(self.project_dir, 'unlabeled-data.json')) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d['example_id'] not in self.project_annotations:
                        continue

                    assert os.path.exists(os.path.join(self.trankit_dir, '{}.json'.format(d['example_id'])))

                    with open(os.path.join(self.trankit_dir, '{}.json'.format(d['example_id']))) as f:
                        inst = json.load(f)

                    self.data.append(inst)

        torch.cuda.empty_cache()
        # print('Loaded {} annotated examples!'.format(len(self.data)))

    def collate_fn(self, batch):
        example_ids = [inst['example_id'] for inst in batch]
        batch_tokens = [inst['tokens'] for inst in batch]
        batch_token_nums = [len(inst['tokens']) for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_lens = []

        max_token_num = max(batch_token_nums)

        batch_labels = []

        for inst in batch:
            token_num = len(inst['tokens'])
            batch_piece_idxs.append(inst['target_piece_idxs'])
            batch_attention_masks.append(inst['target_attention_mask'])
            batch_token_lens.append(inst['target_token_lens'])
            # for identification
            batch_labels.append(inst['label_idxs'] +
                                [0] * (max_token_num - token_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs) if self.config.use_gpu else torch.LongTensor(
            batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks) if self.config.use_gpu else torch.FloatTensor(
            batch_attention_masks)

        batch_labels = torch.cuda.LongTensor(batch_labels) if self.config.use_gpu else torch.LongTensor(batch_labels)
        batch_token_nums = torch.cuda.LongTensor(batch_token_nums) if self.config.use_gpu else torch.LongTensor(
            batch_token_nums)

        batch_distill_mask = torch.cuda.FloatTensor(
            [0 for inst in batch]) if self.config.use_gpu else torch.FloatTensor([0 for inst in batch])

        return TargetBatch(
            example_ids=example_ids,
            texts=[inst['text'] for inst in batch],
            tokens=batch_tokens,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_lens=batch_token_lens,
            label_idxs=batch_labels,
            token_nums=batch_token_nums,
            distill_mask=batch_distill_mask
        )


ALBatch = namedtuple('Batch', field_names=AL_BATCH_FIELDS)


class ALDataset(Dataset):
    def __init__(self, data, config):
        self.max_sent_length = 512
        self.config = config

        self.data = data
        self.pad_id = None

    def batch_num(self, batch_size):
        return len(self.data) // batch_size + int(len(self.data) % batch_size != 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self, xlmr_tokenizer, vocabs):
        self.vocabs = vocabs
        self.pad_id = xlmr_tokenizer.pad_token_id

    def collate_fn(self, batch):
        example_ids = [inst['example_id'] for inst in batch]
        batch_tokens = [inst['tokens'] for inst in batch]
        batch_token_nums = [len(inst['tokens']) for inst in batch]

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_token_lens = []

        for inst in batch:
            if 'piece_idxs' not in inst:  # this happens when target model precomputes distillation signals
                inst['piece_idxs'] = inst['target_piece_idxs']
                inst['attention_mask'] = inst['target_attention_mask']
                inst['token_lens'] = inst['target_token_lens']

        max_piece_num = max([len(inst['piece_idxs']) for inst in batch])
        max_token_num = max(batch_token_nums)

        batch_labels = []

        for inst in batch:
            token_num = len(inst['tokens'])
            batch_piece_idxs.append(inst['piece_idxs'] + [self.pad_id] * (max_piece_num - len(inst['piece_idxs'])))
            batch_attention_masks.append(inst['attention_mask'] + [0] * (max_piece_num - len(inst['piece_idxs'])))
            batch_token_lens.append(inst['token_lens'])
            # for identification
            batch_labels.append(inst['label_idxs'] +
                                [0] * (max_token_num - token_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs) if self.config.use_gpu else torch.LongTensor(
            batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(
            batch_attention_masks) if self.config.use_gpu else torch.FloatTensor(
            batch_attention_masks)

        batch_labels = torch.cuda.LongTensor(batch_labels) if self.config.use_gpu else torch.LongTensor(batch_labels)
        batch_token_nums = torch.cuda.LongTensor(batch_token_nums) if self.config.use_gpu else torch.LongTensor(
            batch_token_nums)

        return ALBatch(
            example_ids=example_ids,
            tokens=batch_tokens,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            token_lens=batch_token_lens,
            labels=[inst['labels'] for inst in batch],
            label_idxs=batch_labels,
            token_nums=batch_token_nums
        )


def precompute_distillation(numberized_data, target_model, config, project_name):
    annotated_set = ALDataset(numberized_data, config)
    annotated_set.vocabs = config.vocabs[project_name]
    annotated_set.pad_id = config.target_tokenizer.pad_token_id
    # test set
    progress = tqdm.tqdm(total=annotated_set.batch_num(config.batch_size), ncols=75,
                         desc='Precomputing distillation signals')

    distil_signals = []
    for batch in DataLoader(annotated_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=annotated_set.collate_fn):
        progress.update(1)
        signals = target_model.compute_distill_signals(batch)
        distil_signals.extend(signals)
    progress.close()

    # print('Computed distillation signals for {} examples'.format(len(distil_signals)))

    with open(os.path.join(DATABASE_DIR, project_name, 'distillations.json'), 'w') as f:
        for distil in distil_signals:
            f.write(json.dumps(distil) + '\n')


def mnlp_sampling(unlabeled_data, model, tokenizer, config, project_id):
    id2example = {}
    for ex in unlabeled_data:
        id2example[ex['example_id']] = ex

    unlabeled_set = ALDataset(unlabeled_data, config)
    unlabeled_set.numberize(tokenizer, model.vocabs)

    progress = tqdm.tqdm(total=unlabeled_set.batch_num(config.batch_size), ncols=75,
                         desc='MNLP:Selecting new examples to annotate')

    selection_pool = []
    for batch in DataLoader(unlabeled_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=unlabeled_set.collate_fn):
        progress.update(1)
        results = model.compute_mnlp_scores(batch)
        selection_pool.extend(
            [{'example_id': res['example_id'], 'informative-level': res['mnlp-score']} for res in results])

    progress.close()
    selection_pool.sort(
        key=lambda x: -x['informative-level'])  # sort in descending order of how informative each example is

    selected_examples = []

    stop = False
    num_exs = 0
    for point in selection_pool:
        ex = id2example[point['example_id']]
        if not stop:
            selected_examples.append(ex)
            num_exs += 1
            if num_exs >= config.num_examples_per_iter:
                stop = True

    return selected_examples


def k_means_clustering(selection_pool, id2example, num_clusters, init_centroids='random', seed=None):
    '''
    Modified from https://github.com/JordanAsh/badge/blob/master/query_strategies/kmeans_sampling.py
    '''
    idxs_unlabeled = np.arange(len(selection_pool))
    embeds_unlabeled = np.array([ex['embedding'] for ex in selection_pool])
    cluster_learner = KMeans(n_clusters=num_clusters, init=init_centroids, random_state=seed)
    cluster_learner.fit(embeds_unlabeled)

    cluster_idxs = cluster_learner.predict(embeds_unlabeled)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embeds_unlabeled - centers) ** 2
    dis = dis.sum(axis=1)
    q_idxs = set(np.array(
        [np.arange(embeds_unlabeled.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in
         range(num_clusters)]).tolist())
    return [id2example[selection_pool[idx]['example_id']] for idx in q_idxs]


def bertkm_sampling(unlabeled_data, model, tokenizer, config, project_id):
    id2example = {}
    for ex in unlabeled_data:
        id2example[ex['example_id']] = ex

    unlabeled_set = ALDataset(unlabeled_data, config)
    unlabeled_set.numberize(tokenizer, model.vocabs)

    progress = tqdm.tqdm(total=unlabeled_set.batch_num(config.batch_size), ncols=75,
                         desc='BERTKM:Selecting new examples to annotate')

    selection_pool = []
    for batch in DataLoader(unlabeled_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=unlabeled_set.collate_fn):
        progress.update(1)
        results = model.compute_embeds_for_bertkm(batch)
        selection_pool.extend(
            [{'example_id': res['example_id'], 'embedding': res['cls-vector']} for res in results])

    progress.close()

    num_clusters = min(config.num_examples_per_iter, len(selection_pool))
    selected_examples = k_means_clustering(selection_pool, id2example, num_clusters, seed=config.seed)

    return selected_examples


def k_means_pp_seeding(selection_pool, id2example, num_clusters, seed):
    '''
    Modified from https://github.com/JordanAsh/badge/blob/master/query_strategies/badge_sampling.py
    '''
    X = np.array([ex['embedding'] for ex in selection_pool])
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    centInds = [0.] * len(X)
    cent = 0
    while len(mu) < num_clusters:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist), seed=seed)
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1

    q_idxs = set(indsAll)
    selected_examples = [id2example[selection_pool[idx]['example_id']] for idx in q_idxs]
    return selected_examples


def badge_sampling(unlabeled_data, model, tokenizer, config, project_id):
    id2example = {}
    for ex in unlabeled_data:
        id2example[ex['example_id']] = ex

    unlabeled_set = ALDataset(unlabeled_data, config)
    unlabeled_set.numberize(tokenizer, model.vocabs)

    progress = tqdm.tqdm(total=unlabeled_set.batch_num(config.batch_size), ncols=75,
                         desc='BADGE:Selecting new examples to annotate')

    selection_pool = []
    for batch in DataLoader(unlabeled_set, batch_size=config.batch_size,
                            shuffle=False, collate_fn=unlabeled_set.collate_fn):
        progress.update(1)
        results = model.compute_embeds_for_badge(batch)
        selection_pool.extend(
            [{'example_id': res['example_id'], 'embedding': res['grad-embed']} for res in results])

    progress.close()

    num_clusters = min(config.num_examples_per_iter, len(selection_pool))
    selected_examples = k_means_pp_seeding(selection_pool, id2example, num_clusters,
                                           seed=config.seed)

    return selected_examples


def random_sampling(unlabeled_data, config):
    full_idxs = list(range(len(unlabeled_data)))
    random.seed(len(full_idxs))
    random.shuffle(full_idxs)

    selected_examples = []

    stop = False
    num_exs = 0

    progress = tqdm.tqdm(total=config.num_examples_per_iter, ncols=75,
                         desc='RANDOM:Selecting new examples to annotate')
    for ex in unlabeled_data:
        if not stop:
            selected_examples.append(ex)
            num_exs += 1
            progress.update(1)
            if num_exs >= config.num_examples_per_iter:
                stop = True
    progress.close()
    return selected_examples
