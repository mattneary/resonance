import itertools
import torch.nn.functional as F
import numpy as np
import torch
from .salience import get_sentences, text_rank, terminal_distr, model

def cos_sim(xs, ys):
    norm_xs = F.normalize(xs, p=2, dim=1)
    norm_ys = F.normalize(ys, p=2, dim=1)
    return xs @ ys.T

def resonance(text_a, text_b):
    sentences_a, ranges_a = get_sentences(text_a)
    sentences_b, ranges_b = get_sentences(text_b)
    adj_a = text_rank(sentences_a)
    adj_b = text_rank(sentences_b)
    salience_a = torch.tensor(terminal_distr(adj_a))
    salience_b = torch.tensor(terminal_distr(adj_b))

    vectors_a = torch.tensor(model.encode(sentences_a))
    vectors_b = torch.tensor(model.encode(sentences_b))
    joint_affinity = cos_sim(vectors_b, vectors_a)

    row_a = salience_a.unsqueeze(0)
    col_b = salience_b.reshape(-1, 1)
    joint_salience = torch.mm(col_b, row_a)

    scores_a = (joint_affinity * joint_salience).T.sum(dim=1)
    print('# Top Sentences from A')
    for sent, score in sorted(zip(sentences_a, scores_a), key=lambda p: p[1])[-5:]:
        print(sent)
    print('# Top Sentences from B')
    scores_b = (joint_affinity * joint_salience).sum(dim=1)
    for sent, score in sorted(zip(sentences_b, scores_b), key=lambda p: p[1])[-5:]:
        print(sent)
