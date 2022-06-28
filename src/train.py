import os
import mindspore
import mindspore.dataset
from model import VQANet, VQALoss
from vqa_set import VQASet
import word2vec
input(
    'Warning: this module should be run on image "mindspore1.7.0-cuda10.1-py3.7-ubuntu18.04", '
    'or you may get tons of errors. '
    'Press Enter to continue.'
)
os.makedirs('checkpoint', exist_ok = True)
class VQACallback(mindspore.train.callback.Callback):
    def __init__(self, val, period_save = 1):
        super().__init__()
        self.val = val
        self.period_save = period_save
    def epoch_end(self, run_context):
        args = run_context.original_args()
        net = args.network.net
        i_epoch = args.cur_epoch_num
        if i_epoch % self.period_save == 0:
            mindspore.save_checkpoint(net, f'checkpoint/{i_epoch}.ckpt')
        accuracy = net.accuracy(self.val)
        print(f'[epoch {i_epoch}] validation accuracy: {accuracy}')
dataset = VQASet(125, 1429, 1429)
net = VQANet()
loss = VQALoss(net)
optimizer = mindspore.nn.SGD(loss.trainable_params())
callback = VQACallback(dataset.val)
model = mindspore.Model(loss, None, optimizer)
model.train(10, dataset.train, callback)
input('Training finished.')
