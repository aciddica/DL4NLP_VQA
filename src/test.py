import mindspore
from model import VQANet
from vqa_set import VQASet
import word2vec
def accuracy(net, dataset):
    n_errors = 0
    n_rows = 0
    for image, question, answer in dataset:
        prediction = net(image, question)
        n_errors += (word2vec.decode_embedding(prediction) - answer).asnumpy().any((1, 2)).sum()
        n_rows += len(answer)
        return 1 - n_errors / n_rows
if __name__ == '__main__':
    net = VQANet()
    dataset_test = VQASet(1, 1429, 1429).test
    i = input('Checkpoint id: ')
    print('Testing...')
    mindspore.load_checkpoint(f'checkpoint/{i}.ckpt', net)
    input(f'Test accuracy: {accuracy(net, dataset_test)}\n')
