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
    all_vectors = torch.tensor(model.encode(sentences_a + sentences_b))
    vectors_a = all_vectors[:len(sentences_a)]
    vectors_b = all_vectors[len(sentences_a):]
    adj_a = text_rank(vectors_a)
    adj_b = text_rank(vectors_b)
    salience_a = torch.tensor(terminal_distr(adj_a))
    salience_b = torch.tensor(terminal_distr(adj_b))

    joint_affinity = cos_sim(vectors_b, vectors_a)

    row_a = salience_a.unsqueeze(0)
    col_b = salience_b.reshape(-1, 1)
    joint_salience = torch.mm(col_b, row_a)

    scores_a = (joint_affinity * joint_salience).T.mean(dim=1)
    scores_b = (joint_affinity * joint_salience).mean(dim=1)
    return scores_a.sum() / len(sentences_a)
