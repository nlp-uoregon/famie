# define model architectures for different tasks here
import torch, os
import torch.nn as nn
from transformers import XLMRobertaModel, AdapterConfig, XLMRobertaTokenizer
import torch.autograd as autograd
from .utils import *


class BaseModel(nn.Module):
    '''
    Modified from https://github.com/nlp-uoregon/trankit/blob/master/trankit/models/base_models.py
    '''

    def __init__(self, config, model_name):
        super().__init__()
        self.config = config
        self.model_name = model_name
        # xlmr encoder
        self.embedding_name = config.proxy_embedding_name if model_name == 'proxy' else config.target_embedding_name
        self.reduction_factor = config.proxy_reduction_factor if model_name == 'proxy' else config.target_reduction_factor

        self.xlmr_dim = EMBEDDING2DIM[self.embedding_name]
        self.xlmr = XLMRobertaModel.from_pretrained(self.embedding_name,
                                                    cache_dir=os.path.join(config.cache_dir, self.embedding_name),
                                                    output_hidden_states=True)
        self.xlmr_dropout = nn.Dropout(p=config.embedding_dropout)
        # add task adapters
        task_config = AdapterConfig.load("pfeiffer",
                                         reduction_factor=self.reduction_factor)
        self.xlmr.add_adapter(model_name, config=task_config)
        self.xlmr.train_adapter([model_name])
        self.xlmr.set_active_adapters([model_name])

    def encode(self, piece_idxs, attention_masks):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]

        wordpiece_reprs = xlmr_outputs[:, 1:-1, :]  # [batch size, max input length - 2, xlmr dim]
        wordpiece_reprs = self.xlmr_dropout(wordpiece_reprs)
        return wordpiece_reprs

    def encode_words(self, piece_idxs, attention_masks, word_lens):
        batch_size, _ = piece_idxs.size()
        all_xlmr_outputs = self.xlmr(piece_idxs, attention_mask=attention_masks)
        xlmr_outputs = all_xlmr_outputs[0]
        cls_reprs = xlmr_outputs[:, 0, :]  # [batch size, xlmr dim]

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = word_lens_to_idxs_fast(word_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.xlmr_dim) + 1
        masks = xlmr_outputs.new(masks).unsqueeze(-1)
        xlmr_outputs = torch.gather(xlmr_outputs, 1,
                                    idxs) * masks
        xlmr_outputs = xlmr_outputs.view(batch_size, token_num, token_len, self.xlmr_dim)
        xlmr_outputs = xlmr_outputs.sum(2)

        xlmr_outputs = self.xlmr_dropout(xlmr_outputs)
        cls_reprs = self.xlmr_dropout(cls_reprs)

        return xlmr_outputs, cls_reprs

    def forward(self, batch):
        raise NotImplementedError


class MultilingualEmbedding(BaseModel):
    '''
        Modified from https://github.com/nlp-uoregon/trankit/blob/master/trankit/models/base_models.py
    '''

    def __init__(self, config, model_name='proxy'):
        super(MultilingualEmbedding, self).__init__(config, model_name=model_name)

    def get_tokenizer_inputs(self, batch):
        wordpiece_reprs = self.encode(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks
        )
        return wordpiece_reprs

    def get_tagger_inputs(self, batch):
        # encoding
        word_reprs, cls_reprs = self.encode_words(
            piece_idxs=batch.piece_idxs,
            attention_masks=batch.attention_masks,
            word_lens=batch.token_lens
        )
        return word_reprs, cls_reprs


class CRF(nn.Module):
    '''
        Modified from https://blender.cs.illinois.edu/software/oneie/
    '''

    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        # self.same_type = self.map_same_types()
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, trn_scr.shape[1]).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens, max_len=logits.shape[1]).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        '''
        Returns: a scalar value = log(P(y1, y2, ..., yn|w)) = sum_{i = 1, n} log(P(y_i|w, y_{i-1:1}))
        '''
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, padded_logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            padded_logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = padded_logits.size()
        vit = padded_logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = padded_logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores


class SeqLabel(nn.Module):
    def __init__(self, config, project_id, model_name):
        super().__init__()
        assert model_name in ['proxy', 'target']
        self.config = config
        self.embedding = MultilingualEmbedding(config, model_name)
        self.xlmr_dim = self.embedding.xlmr_dim

        self.vocabs = config.vocabs[project_id]
        self.label_stoi = self.vocabs['entity-label']  # BIO tags
        self.label_itos = {i: s for s, i in self.label_stoi.items()}
        self.label_num = len(self.label_stoi)

        self.penultimate_ffn = nn.Linear(self.xlmr_dim, config.hidden_num)
        self.label_ffn = nn.Linear(config.hidden_num, self.label_num, bias=False)

        self.crf = CRF(self.label_stoi, bioes=False)

    def forward(self, batch):
        word_reprs, _ = self.embedding.get_tagger_inputs(batch)
        batch_size, _, _ = word_reprs.size()

        h_t = self.penultimate_ffn(word_reprs)
        ori_label_scores = self.label_ffn(h_t)
        label_scores = self.crf.pad_logits(ori_label_scores)
        label_loglik = self.crf.loglik(label_scores,
                                       batch.label_idxs,
                                       batch.token_nums)

        loss = - label_loglik.mean()
        if batch.distill_mask.sum() > 0:
            distill_loss = self.compute_distill_loss(ori_label_scores, batch)

            loss += 0.1 * distill_loss

        return loss

    def predict(self, raw_text):
        tokens = self.config.trankit_tokenizer.tokenize(raw_text, is_sent=True)['tokens']
        piece_idxs, attn_masks, token_lens = subword_tokenize(
            tokens,
            self.config.target_tokenizer,
            self.config.max_sent_length
        )
        batch = TargetBatch(
            example_ids=['raw-text'],
            texts=[raw_text],
            tokens=[tokens],
            piece_idxs=torch.cuda.LongTensor([piece_idxs]) if self.config.use_gpu else torch.LongTensor([piece_idxs]),
            attention_masks=torch.cuda.LongTensor([attn_masks]) if self.config.use_gpu else torch.LongTensor([attn_masks]),
            token_lens=[token_lens],
            label_idxs=[0] * len(tokens),
            token_nums=torch.cuda.LongTensor([len(tokens)]) if self.config.use_gpu else torch.LongTensor([len(tokens)]),
            distill_mask=[0]
        )

        word_reprs, _ = self.embedding.get_tagger_inputs(batch)
        batch_size, _, _ = word_reprs.size()

        h_t = self.penultimate_ffn(word_reprs)
        label_scores = self.label_ffn(h_t)
        label_scores = self.crf.pad_logits(label_scores)
        _, label_pred_ids = self.crf.viterbi_decode(label_scores, batch.token_nums)
        label_pred_ids = label_pred_ids.data.cpu().numpy().tolist()
        label_preds = []
        for bid in range(batch_size):
            pred = [self.label_itos[lid] for lid in label_pred_ids[bid][:batch.token_nums[bid]]]
            label_preds.append(pred)

        return label_preds[0]

    def compute_distill_loss(self, std_lbl_scores, batch):
        '''
        std_lbl_scores.shape = [batch size, seq len, n labels]
        batch.tch_lbl_dist.shape = [batch size, seq len, n labels]
        '''
        batch_size, seq_len, n_labels = std_lbl_scores.shape

        std_lbl_dist = torch.softmax(std_lbl_scores, dim=2).view(batch_size * seq_len, n_labels)
        tch_lbl_dist = batch.tch_lbl_dist.view(batch_size * seq_len, n_labels)

        batch_kl_loss = torch.nn.functional.kl_div(
            input=std_lbl_dist,
            target=tch_lbl_dist,
            reduction='none'
        )  # [batch size * seq len, 1]
        seq_mask = sequence_mask(batch.token_nums, max_len=seq_len) * batch.distill_mask.unsqueeze(-1)  # [bs, seq len]
        seq_mask = seq_mask.view(batch_size * seq_len, 1)
        kl_loss = (batch_kl_loss * seq_mask).sum() / (torch.sum(seq_mask) + 1e-12)

        tch_transitions = batch.transitions * batch.distill_mask.unsqueeze(-1)  # [batch size, 1 + seq len]
        std_transitions = self.crf.calc_binary_score(batch.label_idxs,
                                                     batch.token_nums) * batch.distill_mask.unsqueeze(
            -1)  # [batch size, 1 + seq len]

        transition_loss = torch.sum((tch_transitions - std_transitions) ** 2) / (
                    torch.sum(tch_transitions != 0) + 1e-12)

        return kl_loss * 0.1 + transition_loss

    def compute_distill_signals(self, batch):
        word_reprs, _ = self.embedding.get_tagger_inputs(batch)
        batch_size, _, _ = word_reprs.size()

        h_t = self.penultimate_ffn(word_reprs)
        label_scores = self.label_ffn(h_t)
        normalized_scores = torch.softmax(label_scores, dim=2).data.cpu().numpy().tolist()  # [bs, seq len, n labels]
        batch_transitions = self.crf.calc_binary_score(batch.label_idxs,
                                                       batch.token_nums)  # [batch size, 1 + seq len] this matrix contains transition score for every token
        batch_transitions = batch_transitions.data.cpu().numpy().tolist()
        sent_lens = batch.token_nums.data.cpu().numpy().tolist()

        signals = []
        for bid in range(batch_size):
            actual_len = sent_lens[bid]
            signals.append({
                'example_id': batch.example_ids[bid],
                'signals': {
                    'tch_lbl_dist': normalized_scores[bid][:actual_len],
                    'transitions': batch_transitions[bid][:actual_len] + [0]
                }
            })
        return signals

    def compute_mnlp_scores(self, batch):
        '''
        Implementation of Maximum Normalized Log-Probability (MNLP) proposed in:
            https://openreview.net/pdf?id=ry018WZAZ

        Usage: Higher mnlp score means more informative
        '''
        word_reprs, _ = self.embedding.get_tagger_inputs(batch)
        batch_size, _, _ = word_reprs.size()

        h_t = self.penultimate_ffn(word_reprs)
        label_scores = self.label_ffn(h_t)
        label_scores = self.crf.pad_logits(label_scores)
        _, label_pred_ids = self.crf.viterbi_decode(label_scores, batch.token_nums)
        max_sum_log_P_yi = self.crf.loglik(label_scores,
                                           label_pred_ids,
                                           batch.token_nums)  # [batch size,]
        mnlp_scores = - max_sum_log_P_yi.unsqueeze(-1) / batch.token_nums.unsqueeze(-1)  # [batch size, 1]
        mnlp_scores = mnlp_scores.squeeze(-1).data.cpu().numpy().tolist()  # list of scores

        results = []
        for bid in range(batch_size):
            results.append({
                'example_id': batch.example_ids[bid],
                'mnlp-score': mnlp_scores[bid]
            })

        return results

    def compute_embeds_for_bertkm(self, batch):
        '''
        Implementation of BERT-KM introduced in:
            https://aclanthology.org/2020.emnlp-main.637.pdf
        Usage: output embeddings are used for clustering
        '''
        _, cls_reprs = self.embedding.get_tagger_inputs(batch)
        batch_size = cls_reprs.shape[0]

        cls_reprs = self.penultimate_ffn(cls_reprs)
        cls_reprs = cls_reprs.data.cpu().numpy().tolist()

        results = []
        for bid in range(batch_size):
            results.append({
                'example_id': batch.example_ids[bid],
                'cls-vector': cls_reprs[bid]
            })

        return results

    def compute_embeds_for_badge(self, batch):
        '''
        Implementation of BADGE introduced in:
            https://openreview.net/pdf?id=ryghZJBKPS
            https://aclanthology.org/2020.lifelongnlp-1.1.pdf
        Usage: output embeddings are used for clustering
        '''
        word_reprs, _ = self.embedding.get_tagger_inputs(batch)
        batch_size, _, _ = word_reprs.size()
        seq_mask = sequence_mask(batch.token_nums, max_len=word_reprs.shape[1]).unsqueeze(-1).float()

        h_t = self.penultimate_ffn(word_reprs)
        W_h_t = self.label_ffn(h_t)  # naming follows Eq (1) in https://openreview.net/pdf?id=ryghZJBKPS
        p_t = torch.softmax(W_h_t, dim=2)  # [batch size, seq len, num labels]
        y_hat_t = torch.nn.functional.one_hot(torch.argmax(p_t, dim=2),
                                              num_classes=self.label_num).float()  # [batch size, seq len, num labels]
        # sum_t(p_t[i] - y_hat_t[i]) ht
        g_x = []
        for i in range(self.label_num):
            if i == 0 or self.label_itos[i].startswith('B-'):  # only consider B-* labels to reduce the size
                p_ti = p_t[:, :, i]  # [batch size, seq len]
                y_hat_ti = y_hat_t[:, :, i]  # [batch size, seq len]
                # h_t # [batch size, seq len, rep dim]
                g_x_i = torch.sum((p_ti - y_hat_ti).unsqueeze(-1) * h_t * seq_mask, dim=1)  # [batch size, rep dim]
                g_x.append(g_x_i)

        g_x = torch.cat(g_x, dim=1)  # [batch size, num labels * rep dim]

        grad_embeds = g_x.data.cpu().numpy().tolist()

        results = []
        for bid in range(batch_size):
            results.append({
                'example_id': batch.example_ids[bid],
                'grad-embed': grad_embeds[bid]
            })

        return results
