import mindspore
import mindspore.dataset
class Net(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.lstm = mindspore.nn.LSTM(1, 1)
    def construct(self, x):
        return x * 0
try:
    net = Net()
    optimizer = mindspore.nn.SGD(net.trainable_params())
    model = mindspore.Model(net, None, optimizer)
    dataset = mindspore.dataset.GeneratorDataset([[[1]]], ['1'])
    model.train(1, dataset)
    print('This notebook seems OK.')
else:
    print('This notebook sucks. Switch to another one.')
