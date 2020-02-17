import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sinusoid(max_seq: int, embedding_dim: int):
    return np.array([[
        [
            math.sin(
                pos * math.exp(-math.log(10000) * i / embedding_dim) * math.exp(
                    math.log(10000) / embedding_dim * (i % 2)) + 0.5 * math.pi * (i % 2)
            )
            for i in range(embedding_dim)
        ]
        for pos in range(max_seq)
    ]]).astype(np.float32)


def sinusoid_faster(max_seq: int, embedding_dim: int):
    pos_emb = np.zeros((max_seq, embedding_dim), dtype=np.float32)
    for index in range(0, embedding_dim, 2):
        pos_emb[:, index] = np.array([math.sin(pos / 10000 ** (index / embedding_dim))
                                      for pos in range(max_seq)])
        pos_emb[:, index + 1] = np.array([math.cos(pos / 10000 ** (index / embedding_dim))
                                          for pos in range(max_seq)])
    return pos_emb


def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


class DynamicPositionEmbedding(nn.Module):

    def __init__(self, embedding_dim: int, max_seq=2048):
        super().__init__()
        positional_embedding = sinusoid(max_seq, embedding_dim)  # [1, T, C]
        self.register_buffer('positional_embedding', torch.from_numpy(positional_embedding))

    def forward(self, x: torch.Tensor):
        """

        :param x: [B, T, C]
        :return:
        """
        _B, T, _C = x.shape
        x = x + self.positional_embedding[:, :T, :]
        return x


class RelativeGlobalAttention(nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """

    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h  # num_heads
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb

        # self.E = torch.randn([self.max_seq, int(self.dh)], requires_grad=False)
        E = torch.randn([self.max_seq, self.dh])  # 这个是什么？为什么不是全0？
        self.E = nn.Parameter(E)

        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        # q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.view(q.shape[0], q.shape[1], self.h, -1)
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        # k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.view(k.shape[0], k.shape[1], self.h, -1)
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        # v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.view(v.shape[0], v.shape[1], self.h, -1)
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.shape[2]
        self.len_q = q.shape[2]

        E = self._get_left_embedding(self.len_q, self.len_k)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            # logits += (mask.to(torch.int64) * -1e9).to(logits.dtype)
            logits += mask * -1e9

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        # out = torch.reshape(out, (out.size(0), -1, self.d))
        out = out.view(out.shape[0], -1, self.d)

        out = self.fc(out)
        return out, attention_weights

    def _get_left_embedding(self, len_q, len_k):
        """
        这里没用到len_k，注意一下
        :param len_q:
        :param len_k:
        :return:
        """
        starting_point = max(0, self.max_seq - len_q)
        e = self.E[starting_point:, :]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        # reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        reshaped = padded.view(padded.shape[0], padded.shape[1], padded.shape[-1], padded.shape[-2])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k - self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.shape[-1] - 1, qe.shape[-1] - qe.shape[-2] - 1, -1, device=qe.device),
            qe.shape[-1])
        mask = ~mask
        return mask.to(qe.dtype) * qe


