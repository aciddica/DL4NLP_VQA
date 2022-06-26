import mindspore
class Attention(nn.cell):
    def __init__(self, d = 1024, k = 512, dropout = True):
        super(Attention, self).__init__()