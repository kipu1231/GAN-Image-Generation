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

    # ''' setup random seed '''
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed(args.random_seed)

    ''' load model '''
    print('===> prepare model ...')
    model_gen = models.ACGAN_generator(args)
    model_dcm = models.ACGAN_discriminator(args)

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
        loss_D = 0.0
        loss_G = 0.0
        for idx, data in enumerate(face_loader):
            steps = len(face_loader)
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, iters + 1, steps)
            iters += 1

            ''' TRAIN DISCRIMINATOR '''
            model_dcm.zero_grad()

            ''' prepare real input '''
            img_real, cls = data
            cls = cls.unsqueeze(1).float()
            label_real = torch.full((img_real.size(0),), real_label)

            if torch.cuda.is_available():
                img_real = img_real.cuda()
                cls = cls.cuda()
                label_real = label_real.cuda()

            ''' prepare noise input '''
            in_noise = torch.randn(img_real.size(0), 101, 1, 1)
            fake_cls = torch.randint(0, 2, (img_real.size(0),)).float()
            in_noise[:, -1, 0, 0] = fake_cls
            fake_cls = fake_cls.unsqueeze(1).float()
            label_fake = torch.full((img_real.size(0),), fake_label)

            if torch.cuda.is_available():
                in_noise = in_noise.cuda()
                fake_cls = fake_cls.cuda()
                label_fake = label_fake.cuda()

            ''' loss of real input '''
            y_real_valid, y_real_cls = model_dcm(img_real)

            loss_real_valid = criterion(y_real_valid, label_real)
            loss_real_cls = criterion(y_real_cls, cls)
            loss_real = loss_real_cls + loss_real_valid
            loss_real.backward()

            ''' loss of fake input '''
            img_fake = model_gen(in_noise).detach()
            y_fake_valid, y_fake_cls = model_dcm(img_fake)

            loss_fake_valid = criterion(y_fake_valid, label_fake)
            loss_fake_cls = criterion(y_fake_cls, fake_cls)
            loss_fake = loss_fake_valid + loss_fake_cls
            loss_fake.backward()

            ''' train model '''
            loss_D = loss_fake_valid + loss_real_valid
            #loss_D.backward()
            optimizerD.step()

            train_info += ' loss discriminator: {:.4f}'.format(loss_D.data.cpu().numpy())

            ''' TRAIN GENERATOR '''
            model_gen.zero_grad()

            ''' prepare noise input '''
            in_noise = torch.randn(img_real.size(0), 101, 1, 1)
            fake_cls = torch.randint(0, 2, (img_real.size(0),)).float()
            in_noise[:, -1, 0, 0] = fake_cls
            fake_cls = fake_cls.unsqueeze(1)
            #fake labels are real for generator cost
            label_fake = torch.full((img_real.size(0),), real_label)

            if torch.cuda.is_available():
                in_noise = in_noise.cuda()
                fake_cls = fake_cls.cuda()
                label_fake = label_fake.cuda()

            ''' loss of fake input '''
            img_fake = model_gen(in_noise)
            y_fake_valid, y_fake_cls = model_dcm(img_fake)

            loss_fake_valid = criterion(y_fake_valid, label_fake)
            loss_fake_cls = criterion(y_fake_cls, fake_cls)

            loss_G = loss_fake_valid + loss_fake_cls
            loss_G.backward()
            optimizerG.step()

            ''' print train info '''
            train_info += ' loss generator: {:.4f}'.format(loss_G.data.cpu().numpy())

            print(train_info)

        ''' save model '''
        save_model(model_dcm, os.path.join(args.save_dir, 'acgan_discriminator_model_{}.pth.tar'.format(epoch)))
        save_model(model_gen, os.path.join(args.save_dir, 'acgan_generator_model_{}.pth.tar'.format(epoch)))



