import mindspore.dataset
from image_set import ImageSet
from qa_set import QASet
class VQAPart:
    def __init__(self, images, qas):
        self.images = images
        self.qas = qas
        self.length = len(qas)
    def __getitem__(self, index):
        image_id, question, answer = self.qas[index]
        return self.images[image_id], question, answer
    def __len__(self):
        return self.length
class VQASet:
    def _part(self, part):
        part = VQAPart(self.images, QASet(part))
        part = mindspore.dataset.GeneratorDataset(part, ['image', 'question', 'answer'])
        return part
    def __init__(self, size_batch = 1):
        '''e.g.
        vqa_set = VQASet()
        model.train(n_epochs, vqa_set.train, ...)
        '''
        self.images = ImageSet()
        self.train = self._part('train')
        self.train = self.train.batch(size_batch)
        self.val = self._part('val')
        self.val = self.val.batch(self.val.get_dataset_size())
        self.test = self._part('test')
        self.test = self.test.batch(self.test.get_dataset_size())
