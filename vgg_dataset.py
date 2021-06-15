# vgg_dataset.py

import os
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
    def __init__(self, type, img_size, data_dir):
        self.name2label = {"cats": 0, "dogs": 1, "panda": 2}
        self.img_size = img_size
        self.data_dir = data_dir
        self.data_list = list()
        for file in os.listdir(self.data_dir):
            self.data_list.append(os.path.join(self.data_dir, file))
        print("Load {} Data Successfully!".format(type))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        file = self.data_list[item]
        img = Image.open(file)
        if len(img.split()) == 1:
            img = img.convert("RGB")
        img = img.resize((self.img_size,self.img_size))
        label = self.name2label[os.path.basename(file).split('_')[0]]
        image_to_tensor = ToTensor()
        img = image_to_tensor(img)
        label = tensor(label)
        return img, label

