# vgg_train.py

import os
import time
import torch
import logging
import argparse
import vgg_model, vgg_dataset

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

# train each epoch
def train(epoch, pbar):
    net.train()

    #  start batch
    for step, (images, labels) in pbar:
        if opt.gpu:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_scheduler.step()
        # print(train_scheduler.get_lr()[-1])
        s = ('epoch: %d\t loss: %10s\t lr: %6f\t' % (epoch, loss.item(), train_scheduler.get_last_lr()[-1]))
        pbar.set_description(s)

    # end batch


def val():
    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in test_loader:
        if opt.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct.float() / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss,
        test_acc,
        finish - start
    ))

    return test_loss, test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_path', type=str, default="F:\\Course\\9-Artificial Intelligence\\code\\dataset\\train", help='train images dir')
    parser.add_argument('-test_path', type=str, default="F:\\Course\\9-Artificial Intelligence\\code\\dataset\\test", help='test images dir')
    parser.add_argument('-img_size', type=int, default=128, help='the size of image, mutiple of 32')
    parser.add_argument('-num_class', type=int, default=3, help='the number of class')
    parser.add_argument('-checkpoint_path', type=str, default="checkpoint", help='path to save model')
    parser.add_argument('-batch_size', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-epochs', type=int, default=50, help='epochs')
    parser.add_argument('-milestones', type=float, default=[0.5, 0.8, 0.9], help='milestones')
    parser.add_argument('-gpu', default=True, help='use gpu or not')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate of optimizer')
    parser.add_argument('-tensorboard', default=True, help='use tensorboard or not')

    opt = parser.parse_args()

    # initialize vgg
    if opt.gpu:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class).cuda()
    else:
        net = vgg_model.VGG(img_size=opt.img_size, input_channel=3, num_class=opt.num_class)

    # load data
    train_dataset = vgg_dataset.MyDataset("Train", opt.img_size, opt.train_path)
    test_dataset = vgg_dataset.MyDataset("Test", opt.img_size, opt.test_path)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=2, shuffle=True)  # Reference https://blog.csdn.net/zw__chen/article/details/82806900
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=2, shuffle=True)
    nb = len(train_dataset)

    # Optimzer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.epochs * opt.milestones, gamma=0.1) # learning rate decay

    # checkpoint
    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'epoch_{epoch}-{type}-acc_{acc}.pth')

    # tensorboard
    if opt.tensorboard:
        writer = SummaryWriter(log_dir=opt.checkpoint_path)

    # Start train
    best_acc = 0.0
    for epoch in range(1, opt.epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=int(nb / opt.batch_size))   # process_bar
        train(epoch, pbar)  # train 1 epoch
        loss, acc = val()  # valuation

        if opt.tensorboard:
            writer.add_scalar('Test/Average loss', loss, epoch)
            writer.add_scalar('Test/Accuracy', acc, epoch)

        if epoch > opt.epochs * opt.milestones[1] and best_acc < acc:
            weights_path = checkpoint_path.format(epoch=epoch, type='best', acc=acc)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if epoch == opt.epochs+1:
            weights_path = checkpoint_path.format(epoch=epoch, type='regular', acc=acc)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
    # end epoch
