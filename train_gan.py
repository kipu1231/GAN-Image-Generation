import os
import torch
from tensorboardX import SummaryWriter
from torch import nn
import parser
import models
import data_c
import numpy as np
from torchsummary import summary


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':

    args = parser.arg_parse()

    face_dataset = data_c.Face(args, mode='train')

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    face_loader = torch.utils.data.DataLoader(face_dataset, batch_size=args.train_batch,
                                              num_workers=args.workers, shuffle=True)

    ''' setup GPU '''
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load model '''
    print('===> prepare model ...')
    model_gen = models.GAN_generator(args)
    model_dcm = models.GAN_discriminator(args)

    if torch.cuda.is_available():
        model_gen.cuda()
        model_dcm.cuda()

    model_gen.apply(weights_init)
    model_dcm.apply(weights_init)

    ''' define loss '''
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    ''' setup optimizer '''
    optimizerD = torch.optim.Adam(model_dcm.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(model_gen.parameters(),lr=args.lr, betas=(0.5, 0.999))

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    # Lists to keep track of progress
    img_list = []
    loss_gen = []
    loss_dcm = []
    iters = 0

    ''' Start training '''
    for epoch in range(1, args.epoch + 1):
        iters = 0
        for idx, data in enumerate(face_loader):
            steps = len(face_loader)
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, iters + 1, steps)
            iters += 1

            ''' TRAIN DISCRIMINATOR '''
            ''' train with batch of real data '''
            model_dcm.zero_grad()

            imgs, _ = data

            if torch.cuda.is_available():
                imgs = imgs.cuda()

            batch_size = imgs.size(0)

            label = torch.full((batch_size,), real_label)
            if torch.cuda.is_available():
                label = label.cuda()

            out_real = model_dcm(imgs).view(-1)

            ''' Calculate loss '''
            loss_D_real = criterion(out_real, label)
            loss_D_real.backward()
            #D_x = out.mean().item()

            ''' train with batch of fake data produced by Generator '''
            in_noise = torch.randn(batch_size, 100, 1, 1)
            if torch.cuda.is_available():
                in_noise = in_noise.cuda()

            ''' generate fake images '''
            fake_imgs = model_gen(in_noise)
            label.fill_(fake_label)

            ''' fake images to discriminator '''
            out_fake = model_dcm(fake_imgs.detach()).view(-1)

            loss_D_fake = criterion(out_fake, label)
            loss_D_fake.backward()
            #D_G_z1 = out.mean().item()

            ''' compute overall loss and learn Discriminator '''
            loss_dcm = loss_D_fake + loss_D_real
            #loss_dcm.backward()
            optimizerD.step()

            train_info += ' loss discriminator: {:.4f}'.format(loss_dcm.data.cpu().numpy())

            ''' TRAIN GENERATOR '''
            model_gen.zero_grad()

            in_noise = torch.randn(batch_size*2, 100, 1, 1)
            if torch.cuda.is_available():
                in_noise = in_noise.cuda()

            fake_imgs = model_gen(in_noise)

            label = torch.full((batch_size*2,), real_label)
            if torch.cuda.is_available():
                label = label.cuda()

            #label.fill_(real_label)

            out = model_dcm(fake_imgs).view(-1)

            loss_G = criterion(out, label)
            loss_G.backward()
            optimizerG.step()
            D_G_z2 = out.mean().item()

            ''' write out information to tensorboard '''
            writer.add_scalar('loss generator', loss_G.data.cpu().numpy(), iters)
            train_info += ' loss generator: {:.4f}'.format(loss_G.data.cpu().numpy())

            print(train_info)

        ''' save model '''
        save_model(model_dcm, os.path.join(args.save_dir, 'discriminator_model_{}.pth.tar'.format(epoch)))
        save_model(model_gen, os.path.join(args.save_dir, 'generator_model_{}.pth.tar'.format(epoch)))



