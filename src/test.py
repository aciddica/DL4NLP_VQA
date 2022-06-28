import mindspore
from model import VQANet
from vqa_set import VQASet
import word2vec
i = input('Checkpoint id: ')
print('Testing...')
net = VQANet()
dataset_test = VQASet(1, 1429, 1429).test
mindspore.load_checkpoint(f'checkpoint/{i}.ckpt', net)
input(f'Test accuracy: {net.accuracy(dataset_test)}')
