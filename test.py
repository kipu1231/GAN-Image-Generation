import matplotlib
import torch
from PIL import Image
import parser
import models
import data_c
import numpy as np
import os
import torchvision.utils as utils


if __name__ == '__main__':
    args = parser.arg_parse()

    np.random.seed(100)
    torch.manual_seed(100)

    ''' prepare model '''
    generator_p1 = models.GAN_generator(args)
    generator_p2 = models.ACGAN_generator(args)

    if torch.cuda.is_available():
        generator_p1.cuda()
        generator_p2.cuda()

    checkpoint_p1 = torch.load(args.resume1)
    #checkpoint_p1 = torch.load('generator_model.pth.tar',  map_location = 'cpu')
    generator_p1.load_state_dict(checkpoint_p1)
    generator_p1.eval()

    checkpoint_p2 = torch.load(args.resume2)
    #checkpoint_p2 = torch.load('acgan_generator_model_15.pth.tar',  map_location = 'cpu')
    generator_p2.load_state_dict(checkpoint_p2)
    generator_p2.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    file_name_p1 = os.path.join(args.save_dir, "fig_gan.jpg")
    file_name_p2 = os.path.join(args.save_dir, "fig_acgan.jpg")

    in_noise1 = torch.randn(32, 100, 1, 1)
    in_noise2a = torch.randn(10, 101, 1, 1)
    help = in_noise2a
    #in_noise2b = torch.randn(10, 101, 1, 1)

    label_2a = torch.full((10,), 0).float()
    label_2b = torch.full((10,), 1).float()

    in_noise2a[:, -1, 0, 0] = label_2a
    help[:, -1, 0, 0] = label_2b

    in_noise2 = torch.cat((in_noise2a, help))

    if torch.cuda.is_available():
        in_noise2 = in_noise2.cuda()
        in_noise1 = in_noise1.cuda()

    with torch.no_grad():
        imgs = generator_p1(in_noise1).detach()
        imgs2 = generator_p2(in_noise2).detach()

    utils.save_image(imgs.data[:32], file_name_p1, nrow=8, normalize=True)
    utils.save_image(imgs2.data[:20], file_name_p2, nrow=10, normalize=True)









