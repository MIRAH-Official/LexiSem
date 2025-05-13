import numpy as np
from bart_score import BARTScorer
from bert_score import score
from os.path import join
import os
import argparse

"""
    Code from SummaReranker (ACL, 2022)
    - link : https://github.com/Ravoxsg/SummaReranker-ACL-22/blob/main/src/common/evaluation.py
"""
def bartscore_eval(val_summaries, val_labels, verbose=True):
    print("\n", "*" * 10, "3 - BARTScore evaluation", "*" * 10)
    bart_scorer = BARTScorer(device = "cuda:1", checkpoint = 'facebook/bart-large-cnn')
    bartscore_scores = bart_scorer.score(val_labels, val_summaries)
    m_bartscore = np.mean(np.array(bartscore_scores))
    print("Mean BARTScore: {:.2f}".format(m_bartscore))
    return np.array(bartscore_scores)

def bertscore_eval(val_summaries, val_labels, verbose=True):
    print("\n", "*" * 10, "2 - BERTScore evaluation", "*" * 10)
    p, r, f1 = score(val_summaries, val_labels, lang='en', verbose=verbose)
    mean_f1 = 100 * f1.mean()
    print("Mean BERTScore F1: {:.2f}".format(mean_f1))
    return 100 * f1.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--ref", type=str, help="path of a directory or a file containing reference summaries", required=True)
    parser.add_argument("--hyp", type=str, help="path of a directory or a file containing candidate summaries", required=True)
    args = parser.parse_args()

    ref_dir = args.ref
    hyp_dir = args.hyp

    # calculate bertscore
    cnt = 0
    num = len(os.listdir(ref_dir))
    print("num : ",num)
    ref_list, hyp_list = [], []
    for i in range(num):
        ref = open(join(ref_dir, f"{i}.ref"), 'r',encoding='utf-8').read()
        ref_list.append(ref)

        hyp = open(join(hyp_dir, f"{i}.dec"), 'r',encoding='utf-8').read()
        hyp_list.append(hyp)
        # print(len(hyp_list))

    bartscore_eval(hyp_list, ref_list)
    