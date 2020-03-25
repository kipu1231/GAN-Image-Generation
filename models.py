import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamb

        return output, None


def grad_reverse(x, lamb):
    return GradientReverse.apply(x, lamb)


class GAN_generator(nn.Module):
    def __init__(self, args):
        super(GAN_generator, self).__init__()

        self.generator = nn.Sequential(
            # k-2p=2
            # 100*1*1 --> 512*4*4
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            # 512*4*4 --> 256*8*8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5, True),
            nn.ReLU(True),

            # 256*8*8 --> 128*16*16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),

            # 128*16*16 --> 64*32*32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            #nn.Dropout2d(0.5, True),
            nn.ReLU(True),

            # Generator output won't use BN.
            # 64*32*32 --> 3*64*64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, img):
        x = self.generator(img)
        return x


class GAN_discriminator(nn.Module):
    def __init__(self, args):
        super(GAN_discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # k-2p=2
            # 3x64x64 --> 64x32x32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1,bias=False),
            nn.LeakyReLU(0.2, True),

            # 64x32x32 --> 128x16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True),

            # 128x16x16 --> 256x8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True),

            # 256x8x8 --> 512x4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4,stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.discriminator(img)
        return x


class ACGAN_generator(nn.Module):
    def __init__(self, args):
        super(ACGAN_generator, self).__init__()

        self.generator = nn.Sequential(
            # k-2p=2
            # 101*1*1 --> 512*4*4
            nn.ConvTranspose2d(101, 1024, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 512*4*4 --> 256*8*8
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 256*8*8 --> 128*16*16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 128*16*16 --> 64*32*32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Generator output won't use BN.
            # 64*32*32 --> 3*64*64
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, img):
        x = self.generator(img)
        return x


class ACGAN_discriminator(nn.Module):
    def __init__(self, args):
        super(ACGAN_discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # k-2p=2
            # 3x64x64 --> 64x32x32
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1,bias=False),
            nn.LeakyReLU(0.2, True),

            # 64x32x32 --> 128x16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            # 128x16x16 --> 256x8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            # 256x8x8 --> 512x4x4
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True)
        )
        self.rf = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )

        self.cls = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.discriminator(img)
        #x = x.view(-1, 1024)

        valid = self.rf(x)
        valid = valid.view(valid.size(0), -1)

        cls = self.cls(x)
        cls = cls.view(cls.size(0), -1)

        return valid, cls


