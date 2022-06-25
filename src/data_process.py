import os
import json
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as vision
from PIL import Image
from easydict import EasyDict as edict

class VQSdataset:
    def __init__(self, images, mode = "train", Q_A_df = None):
        # self.imgdir_path = ""
        # self.annotations = {}
        # self.qusetion = {}
        self.Q_A_df = Q_A_df
        # self.Resize = vision.Resize((512, 512))
        # self.CenterCrop =  vision.CenterCrop(448)
        self.images = images
        # print (os.getcwd())
        # if mode == "train":
        #     self.imgdir_path = "./data/images/train"
        #     self.annotations = json.load(open("./data/annotations/train.json"))
        #     self.qusetion = json.load(open("./data/questions/train.json"))
        # elif mode == "test":
        #     self.imgdir_path = "./data/images/test"
        #     self.annotations = json.load(open("./data/annotations/test.json"))
        #     self.qusetion = json.load(open("./data/questions/test.json"))
        # else:
        #     self.imgdir_path = "./data/images/val"
        #     self.annotations = json.load(open("./data/annotations/val.json"))
        #     self.qusetion = json.load(open("./data/questions/val.json"))
        # img_count = 0
        # for index, item in enumerate(self.annotations['annotations']):
        #     img_path = os.path.join(self.imgdir_path, "COCO_{0}2014_{1}.jpg".format(mode,str(item['image_id']).zfill(12))) 
        #     if os.path.exists(img_path):
        #         img_count += 1
        #         continue
        #     self.images.append(img_path)
        
    def __getitem__(self, index):
        return self.images[self.Q_A_df['image_id'][index]], self.Q_A_df['question_tensor'][index], self.Q_A_df['annotation'][index]

    def __len__(self):
        return len(self.Q_A_df['question'])

# def img2tensor(img):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     transform = edict({
#         "ToPIL": vision.ToPIL(),
#         "Decode": vision.Decode(),
#         "Resize": vision.Resize((512, 512)),
#         "CenterCrop": vision.CenterCrop(448),
#         "ToTensor":vision.ToTensor(),
#         "Normalize": vision.Normalize(mean=mean, std=std),
#         "HWC2CHW": vision.HWC2CHW(),
#     })
#     img = transform.HWC2CHW(img) / 255
#     img = transform.Normalize(img)
#     return img

class DistributedSampler():
    """
    sampling the dataset.
    Args:
    Returns:
        num_samples, number of samples.
    """
    def __init__(self, dataset, rank=0, group_size=1, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(self.dataset_length * 1.0 / self.group_size)
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset_length)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples
        
def create_dataset(batch_size, images, Q_A_df = None, mode = 'train', drop_remainder = True):
    
    raw_dataset = VQSdataset(mode, images, Q_A_df)
    sampler = DistributedSampler(raw_dataset)
    dataset = ds.GeneratorDataset(raw_dataset, ["image", "question", "label"], shuffle = False, sampler = sampler)
    # dataset = dataset.map(operations = img2tensor, input_columns = "image", num_parallel_workers = 8)
    dataset = dataset.batch(batch_size, drop_remainder = drop_remainder)

    print("creating dataset...")
    print(dataset)

    return dataset