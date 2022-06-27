import PIL.Image
import json
import numpy
import os
import re
import mindspore
path_data = 'data'
image_id_max = 600000
i_image_max = 80000
size_image_in_memory_max = 256
class ImageSet:
    @staticmethod
    def process(size_image):
        os.makedirs(f'image_set_{size_image}', exist_ok = True)
        in_memory = size_image < size_image_in_memory_max
        if in_memory:
            i_image = 0
            image_ids = numpy.empty((i_image_max,), numpy.uint32)
            images = numpy.empty((i_image_max, 3, size_image, size_image), numpy.uint8)
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
                        image = image.transpose(2, 0, 1)
                        if in_memory:
                            image_ids[i_image] = image_id
                            images[i_image] = image
                            i_image += 1
                        else:
                            image = numpy.ascontiguousarray(image)
                            with open(f'image_set_{size_image}/{image_id}', 'wb') as f:
                                numpy.save(f, image)
        if in_memory:
            with open(f'image_set_{size_image}/image_ids', 'wb') as f:
                numpy.save(f, image_ids[:i_image])
            with open(f'image_set_{size_image}/images', 'wb') as f:
                numpy.save(f, images[:i_image])
    def __init__(self, size_image = 224):
        '''e.g.
        images = ImageSet(64)
        images[i] # returns numpy.ndarray in shape 3, 64, 64
        '''
        if not os.path.exists(f'image_set_{size_image}'):
            self.process(size_image)
        self.size_image = size_image
        self.in_memory = size_image < size_image_in_memory_max
        self.empty = numpy.zeros((3, size_image, size_image), numpy.float32)
        self.index = [self.empty] * image_id_max
        if self.in_memory:
            with open(f'image_set_{self.size_image}/image_ids', 'rb') as f:
                image_ids = numpy.load(f)
            with open(f'image_set_{self.size_image}/images', 'rb') as f:
                images = numpy.load(f)
            for image_id, image in zip(image_ids, images):
                self.index[image_id] = image
        else:
            for i in os.listdir(f'image_set_{size_image}'):
                self.index[int(i)] = None
    def __getitem__(self, index):
        '''e.g.
        images[25] # returns numpy.ndarray in shape 3, 64, 64
        images[0] # no such image -> returns numpy.zeros((3, 64, 64))
        # pixel values are of type numpy.float32 and in range [0, 1)
        '''
        image = self.index[index]
        if image is None:
            with open(f'image_set_{self.size_image}/{index}', 'rb') as f:
                image = numpy.load(f)
        return image.astype(numpy.float32) / 256
