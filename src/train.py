import metric
from model import VQANet, Loss
from vqa_set import VQASet
class VQACallback(mindspore.train.callback.Callback):
    def __init__(self, val, period_save = 1):
        super().__init__()
        self.val = val
    def step_end(self, run_context):
        args = run_context.original_args()
        net = args.network.net
        i_epoch = args.cur_epoch_num
        if i_epoch % period_save == 0:
            mindspore.save_checkpoint(net, f'checkpoint/{i_epoch}.ckpt')
        image, question, answer = next(iter(self.val))
        accuracy = metric.accuracy(net(image, question), answer)
        print(f'[epoch {i_epoch}] loss: {args.net_outputs.asnumpy():f}, validation accuracy: {accuracy}')
dataset = VQASet()
net = VQANet(224, 8, 100, 1024, 1024)
loss = VQALoss(net)
optimizer = mindspore.nn.SGD(loss.trainable_params())
callback = VQACallback(dataset.val)
model = mindspore.Model(loss, None, optimizer)
model.train(10, dataset.train, callback)
