import mindspore.nn as nn
import mindspore.ops as ops
class Attention(nn.cell):
    def __init__(self, d = 1024, k = 512, dropout = True):
        super(Attention, self).__init__()
        self.ff_image = nn.Dense(d, k)
        self.ff_ques = nn.Dense(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Dense(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq)
        expand_dims = ops.ExpandDims()
        hq = expand_dims(hq, 1)
        # N * 196 * 512
        ha = nn.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha)
        squeeze = ops.Squeeze(2)
        ha = squeeze(ha)
        pi = nn.Softmax(ha)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (expand_dims(pi, 2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u