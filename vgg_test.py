# vgg_test.py

import time
import torch
import argparse
import vgg_model, vgg_dataset

from torch.utils.data import DataLoader


def val(test_loader):
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        if opt.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    test_acc = correct.float() / len(test_loader.dataset)

    print('Test set: Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_acc,
        finish - start
    ))

    return test_loss, test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', type=str, default="F:\\Course\\9-Artificial Intelligence\\code\\dataset\\test", help='images path')
    parser.add_argument('-model_path', type=str, default="checkpoint\\epoch_41-best-acc_0.8600000143051147.pth", help='model path')
    parser.add_argument('-img_size', type=int, default=128, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=3, help='the number of class')
    parser.add_argument('-gpu', default=True, help='use gpu or not')

    opt = parser.parse_args()

    # initialize vgg
    if opt.gpu:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class)

    # load model data
    net.load_state_dict(torch.load(opt.model_path))
    net.eval()

    test_dataset = vgg_dataset.MyDataset("Test", opt.img_size, opt.img_path)
    test_loader = DataLoader(test_dataset, shuffle=True)

    val(test_loader)



