from torch.utils.data import Dataset
import os
import json
import torch
from transformers import RobertaTokenizer, AutoTokenizer,BartTokenizer
from torch.nn.utils.rnn import pad_sequence


def to_cuda(batch, gpuid):
    for n in batch:
        if n != "data" and not isinstance(batch[n], list):
            batch[n] = batch[n].to(gpuid)

def collate_mp(batch, pad_token_id, is_test=False):

    result = {}
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result
    

    def pad_2d_tensor_list(tensor_list):
        # tensor_list: list of [num_sents, seq_len] tensors
        max_sents = max(x.size(0) for x in tensor_list)
        max_len = tensor_list[0].size(1)
        padded = []
        for t in tensor_list:
            if t.size(0) < max_sents:
                pad = torch.full((max_sents - t.size(0), max_len), pad_token_id, dtype=t.dtype)
                t = torch.cat([t, pad], dim=0)
            padded.append(t)
        return torch.stack(padded)  # [batch, max_sents, seq_len]

    def pad_3d_tensor_list(list_of_2d_tensors):
        # list of [num_candidates, num_sents, seq_len]
        max_candidates = max(x.size(0) for x in list_of_2d_tensors)
        max_sents = max(x.size(1) for x in list_of_2d_tensors)
        seq_len = list_of_2d_tensors[0].size(2)
        padded = []
        for cand in list_of_2d_tensors:
            pad_cands = cand
            if cand.size(0) < max_candidates:
                pad = torch.full((max_candidates - cand.size(0), cand.size(1), cand.size(2)), pad_token_id, dtype=cand.dtype)
                pad_cands = torch.cat([cand, pad], dim=0)
            if pad_cands.size(1) < max_sents:
                pad = torch.full((pad_cands.size(0), max_sents - pad_cands.size(1), seq_len), pad_token_id, dtype=cand.dtype)
                pad_cands = torch.cat([pad_cands, pad], dim=1)
            padded.append(pad_cands)
        return torch.stack(padded)  # [batch, max_candidates, max_sents, seq_len]

    # === Base fields ===
    src_input_ids = pad([x['src_input_ids'] for x in batch])

    result['src_input_ids'] = src_input_ids

    # === Abstract IDs ===
    result['abstract_ids'] = pad_2d_tensor_list([x['abstract_ids'] for x in batch])  # [batch, max_sents, seq_len]

    # === Candidate IDs ===
    result['candidate_ids'] = [x['candidate_ids'] for x in batch]       # [batch, num_cands, max_sents, seq_len]

    # === Raw text for semantic scoring ===
    result['abstract'] = [x['abstract'] for x in batch]
    result['candidates'] = [x['candidates'] for x in batch]

    # === Optional training fields ===
    if is_test:
        result['data'] = [x['data'] for x in batch] if 'data' in batch[0] else batch
    else:   # train
        # positive weights
        pos_weights = torch.stack([x['positive_weights'] for x in batch])
        result['positive_weights'] = pos_weights

        # costs
        costs = torch.stack([x["costs"] for x in batch])
        result['costs'] = costs

        # negative
        negative_ids = [x['negative_ids'] for x in batch]
        max_len = max([max([len(c) for c in x]) for x in negative_ids])
        negative_ids = [pad(x, max_len) for x in negative_ids]
        result['negative_ids'] = torch.stack(negative_ids)

    return result


class SumDataset(Dataset):
    def __init__(self, fdir, model_type,tokenizer, max_len=-1, is_test=False, total_len=512, is_sorted=True, max_num=-1, is_untok=True, num=-1, neg_size=16, thre=0):
        """ dataformat : article, reference, [(candidate_i, score_i)]"""
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        
        self.tok = AutoTokenizer.from_pretrained(tokenizer, verbose=False)#

        self.maxlen = max_len       # candidate max length
        self.maxnum = max_num       # candidate num
        self.is_test = is_test      # only evaluate
        self.total_len = total_len  # document max length
        self.sorted = is_sorted     
        self.is_untok = is_untok
        self.neg_size = neg_size    # negative num
        self.thre = thre


    def __len__(self):
        return self.num
    

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)

        if self.is_untok:
            article = data['article_untok']
            abstract = data['abstract_untok']
        else:
            article = data['article']
            abstract = data['abstract']

        # === Document ===
        cls_token = self.tok.cls_token
        src_txt = cls_token.join(article)
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors='pt', pad_to_max_length=False, truncation=True)
        src_input_ids = src['input_ids']
        src_input_ids = src_input_ids.squeeze(0)
        ##########start_change
        # === Abstract (sentence-level) ===
        abstract_enc = self.tok(
            abstract, padding="max_length", truncation=True, max_length=self.maxlen, return_tensors="pt"
        )
        abstract_ids = abstract_enc["input_ids"]  # [num_sents, seq_len]
        ###########end_change
        
        # === Candidates ===
        candidates = data['candidates_untok']
        _candidates = data['candidates']
        data['candidates'] = _candidates

        if self.sorted:  # training mode
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x: x[1], reverse=True)
            data['candidates'] = _candidates

        if self.maxnum > 0:
            candidates = candidates[:self.maxnum]
            _candidates = _candidates[:self.maxnum]

        if not self.is_untok:
            candidates = _candidates
        
        ##########start_change
        candidate_ids = []
        for x in candidates:
            sentences = x[0]  # list of sentences
            enc = self.tok(sentences, padding="max_length", truncation=True, max_length=self.maxlen, return_tensors="pt")
            candidate_ids.append(enc["input_ids"])  # [num_sents, seq_len]
        ###########end_change
        result = {
            "src_input_ids": src_input_ids,
            "candidate_ids": candidate_ids,  # List of [num_sents, seq_len]
            "abstract_ids": abstract_ids,    # [num_sents, seq_len]
            "abstract": abstract,            # raw text
            "candidates": [x[0] for x in candidates]  # raw text
        }

        if self.is_test:
            result['data'] = data
        else:   # train
            # positive weights
            pos_weights = data[str(self.thre)][:self.maxnum]
            result['positive_weights'] = torch.FloatTensor(pos_weights)

            # costs
            costs = torch.FloatTensor([(1-x[1]) for x in candidates])
            result['costs'] = costs

            # negative ids
            negatives = data['negative_untok']
            negatives = negatives[:self.neg_size]
            neg_txt = [" ".join(x) for x in negatives]

            neg = self.tok.batch_encode_plus(neg_txt, max_length=self.maxlen, return_tensors='pt', pad_to_max_length=False, truncation=True, padding=True)
            result['negative_ids'] = neg['input_ids']

        return result
