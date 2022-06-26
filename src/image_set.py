import PIL.Image
import json
import numpy
import os
import re
import mindspore
path_data = 'data'
class ImageSet:
    @staticmethod
    def process(size_image):
        os.makedirs(f'image_set_{size_image}', exist_ok = True)
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
                        image = image.resize((size_image, size_image))
                        image = image.convert('RGB')
                        image = numpy.array(image)
                        image = numpy.ascontiguousarray(image.transpose(2, 0, 1))
                        image = image.reshape((1,) + image.shape)
                        with open(f'image_set_{size_image}/{image_id}', 'wb') as f:
                            numpy.save(f, image)
    def _load_image(self, index):
        with open(f'image_set_{self.size_image}/{index}', 'rb') as f:
            return numpy.load(f)
    def __init__(self, size_image = 224):
        '''e.g.
        images = ImageSet(64)
        images[i] # returns numpy.ndarray in shape 1, 3, 64, 64
        '''
        if not os.path.exists(f'image_set_{size_image}'):
            self.process(size_image)
        self.size_image = size_image
        self.in_memory = size_image < 256
        self.empty = numpy.zeros((1, 3, size_image, size_image), numpy.float32)
        self.image_list = [self.empty] * 600000
        for i in os.listdir(f'image_set_{size_image}'):
            self.image_list[int(i)] = self.in_memory and self._load_image(i)
    def __getitem__(self, index):
        '''e.g.
        images[25] # returns numpy.ndarray in shape 1, 3, 64, 64
        images[0] # no such image -> returns numpy.zeros((1, 3, 64, 64))
        # pixel values are of type numpy.float32 and in range [0, 1)
        '''
        image = self.image_list[index]
        if image is False:
            image = self._load_image(index)
        return image.astype(numpy.float32) / 256
