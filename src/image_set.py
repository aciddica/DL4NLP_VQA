'''
因为val和test中图片有重复，考虑建一个大图片池，给不同数据集分别引用，如：
class DataSet:
    def __init__(self, images, ...):
        self.images = images
        ...
images = ImageSet(128)
dataset_train = DataSet(images, ...)
dataset_val = DataSet(images, ...)
...
访问不存在的下标会返回零张量，这样没图的问题也可以正常参与训练了。
'''
import PIL.Image
import json
import numpy
import os
import re
import mindspore
path_data = 'data'
class ImageSet:
    @staticmethod
    def process(image_size):
        os.makedirs(f'image_set_{image_size}', exist_ok = True)
        for part in 'train', 'val', 'test':
            if part != 'test':
                with open(f'{path_data}/questions/{part}.json') as f:
                    image_id_set = set(i['image_id'] for i in json.load(f)['questions'])
            for name in os.listdir(f'{path_data}/images/{part}'):
                image_id = int(re.match(r'COCO_(train|val)2014_(\d+).jpg', name).group(2))
                if (part == 'test') ^ (image_id in image_id_set):
                    with PIL.Image.open(f'{path_data}/images/{part}/{name}') as image:
                        h = image.height
                        w = image.width
                        m = max(h, w)
                        image = image.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
                        image = image.resize((image_size, image_size))
                        image = image.convert('RGB')
                        image = numpy.array(image)
                        image = numpy.ascontiguousarray(image.transpose(2, 0, 1))
                        with open(f'image_set_{image_size}/{image_id}', 'wb') as f:
                            numpy.save(f, image)
    def _load_image(self, index):
        with open(f'image_set_{self.image_size}/{index}', 'rb') as f:
            return numpy.load(f)
    def __init__(self, image_size, dtype = mindspore.float64):
        '''e.g.
        images = ImageSet(64)
        images[i] # returns mindspore.Tensor of shape 3, 64, 64
        '''
        if not os.path.exists(f'image_set_{image_size}'):
            self.process(image_size)
        self.image_size = image_size
        self.dtype = dtype
        self.in_memory = image_size < 256
        self.empty = numpy.zeros((3, self.image_size, self.image_size))
        self.image_list = [self.empty] * 600000
        for i in os.listdir(f'image_set_{self.image_size}'):
            self.image_list[int(i)] = self.in_memory and self._load_image(i)
    def __getitem__(self, index):
        '''e.g.
        images[25] # returns mindspore.Tensor of shape 3, 64, 64
        images[0] # returns zero tensor of the shape (no such image)
        '''
        image = self.image_list[index]
        if image is False:
            image = self._load_image(index)
        return mindspore.Tensor(image, self.dtype)
