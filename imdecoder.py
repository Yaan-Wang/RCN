import torch
from torch import nn
import torch.nn.functional as F
# from vgg19 import VGG19
# backbone_nets = {'vgg19': VGG19}


class Decoder(nn.Module):
    def __init__(self, image_channel_size, conv_channel_size):
        super(Decoder, self).__init__()
        self.image_channel_size = image_channel_size
        self.conv_channel_size = conv_channel_size

        self.deconv1 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*8,
                                          out_channels=self.conv_channel_size*4,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn1 = nn.BatchNorm2d(num_features=self.conv_channel_size*4,)

        self.deconv2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size*4,
                                          out_channels=self.conv_channel_size*2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.bn2 = nn.BatchNorm2d(num_features=self.conv_channel_size*2,)

        self.deconv2_2 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 2,
                                          out_channels=self.conv_channel_size * 2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn2_2 = nn.BatchNorm2d(num_features=self.conv_channel_size * 2, )

        self.deconv3 = nn.ConvTranspose2d(in_channels=self.conv_channel_size * 2,
                                          out_channels=self.conv_channel_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                          )

        self.bn3 = nn.BatchNorm2d(num_features=self.conv_channel_size, )

        self.deconv6 = nn.ConvTranspose2d(in_channels=self.conv_channel_size,
                                          out_channels=self.image_channel_size,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=1,
                                         )

        self.relu = nn.ReLU(inplace=True)

        self.deconv7 = nn.Conv2d(1440,self.conv_channel_size*2 , kernel_size=1, stride=1, padding=0)
        self.project = nn.Sequential(
            nn.Conv2d(4*self.conv_channel_size, 2*self.conv_channel_size, 1, bias=False),
            nn.BatchNorm2d(2*self.conv_channel_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x,latent_x):

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)

        # fuse
        latent_x = self.deconv7(latent_x)
        latent_x = self.bn2(latent_x)
        latent_x = self.relu(latent_x)
        latent_x = F.interpolate(latent_x, size=x.shape[-2:], mode='bilinear', align_corners=False)
        res = torch.cat((latent_x, x), dim=1)
        res = self.project(res)

        x = self.deconv3(res)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv6(x)

        return x
