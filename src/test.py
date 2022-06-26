import metric
from model import VQANet
from vqa_set import VQASet
dataset_test = VQASet().test
net = VQANet()
i = input('checkpoint id: ')
print('testing...')
mindspore.load_checkpoint(f'checkpoint/{i}.ckpt', net)
image, question, answer = next(iter(dataset.test))
accuracy = metric.accuracy(net(image, question), answer)
input(f'test accuracy: {accuracy}\n')
