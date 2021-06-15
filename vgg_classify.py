# vgg_classify.py

import torch
import argparse
import vgg_model

from PIL import Image
from torchvision.transforms import ToTensor

class_names = ["cats", "dogs", "panda"]


class Classify:
    def __init__(self, model_path, img_size, num_class, use_gpu=True):
        self.model_path = model_path
        self.img_size = img_size
        self.num_class = num_class
        self.use_gpu = use_gpu
        self.init_model()

    def init_model(self):
        # initialize model
        if self.use_gpu:
            self.net = vgg_model.VGG(img_size=self.img_size, input_channel=3, num_class=self.num_class).cuda()
        else:
            self.net = vgg_model.VGG(img_size=self.img_size, input_channel=3, num_class=self.num_class)

        # load model data
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.eval()

    def classify(self, image_path):
        img = Image.open(image_path)
        if len(img.split()) == 1:
            img = img.convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        image_to_tensor = ToTensor()
        img = image_to_tensor(img)
        img = img.unsqueeze(0)
        if self.use_gpu:
            img = img.cuda()
        output = self.net(img)

        _, indices = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        precision = percentage[int(indices)].item()
        result = class_names[indices]
        print('Precision',precision)
        print('Rredicted:', result)
        return precision, result
        # https://blog.csdn.net/qq_41167777/article/details/109013155


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="F:\\Course\\9-Artificial Intelligence\\code\\dataset\\image\\dog.jpg", help='images path')
    parser.add_argument('-model_path', type=str, default="checkpoint\\epoch_41-best-acc_0.8600000143051147.pth", help='model path')
    parser.add_argument('-img_size', type=int, default=128, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=3, help='the number of class')
    parser.add_argument('-gpu', default=True, help='use gpu or not')

    opt = parser.parse_args()

    classify = Classify(opt.model_path,opt.img_size,opt.num_class,opt.gpu)
    classify.init_model()
    classify.classify(opt.img_path)



