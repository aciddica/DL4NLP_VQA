import mindspore.nn as nn
import mindspore.ops as ops
class Attention(nn.Cell):
    def __init__(self, d = 1024, k = 512, dropout = True):
        super(Attention, self).__init__()
        self.ff_image = nn.Dense(d, k)
        self.ff_ques = nn.Dense(d, k)
        if dropout:
            self.dropout = nn.Dropout(keep_prob=0.5)
        self.ff_attention = nn.Dense(k, 1)

    def construct(self, vi, vq):
        hi = self.ff_image(vi)
        hq = self.ff_ques(vq)
        expand_dims = ops.ExpandDims()
        hq = expand_dims(hq, 1)
        ha = nn.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.Dropout(ha)
        ha = self.ff_attention(ha)
        squeeze = ops.Squeeze(2)
        ha = squeeze(ha)
        pi = nn.Softmax(ha)
        vi_attended = (expand_dims(pi, 2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u
