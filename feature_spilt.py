import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vgg19 import VGG19
from swinbackbone import SwinTransformer
from rbfmodel import BasicRFB_a
from deforconvblock import ConvOffset2D

# from swin_transformer import  build_s
# backbone nets
#backbone_nets = {'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19}
backbone_nets = {'vgg19': VGG19}

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, channel,kernel_size, output_size=None, dilation=1, stride=1, device=torch.device('cpu')):
        super(AvgFeatAGG2d, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size
        self.S=4
        conv_kernels = [1, 1, 1, 1]
        conv_groups =  [1, 1, 1, 1]#[1, 2, 4, 8]

        self.conv_1 = conv(channel, channel // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=1, groups=conv_groups[0])
        self.conv_2 = conv(channel, channel // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=1, groups=conv_groups[1])
        self.conv_3 = conv(channel, channel // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=1, groups=conv_groups[2])
        self.conv_4 = conv(channel, channel // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=1, groups=conv_groups[3])
        # self.convs = []
        # for i in range(4):
        #     self.convs.append(nn.Conv2d(channel // self.S, channel //  self.S, kernel_size=2 * (i + 1) + 1, padding=i + 1))


    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        # print( input.size())
        # print(input.size())
        x1 = self.conv_1(input)
        x2 = self.conv_2(input)
        x3 = self.conv_3(input)
        x4 = self.conv_4(input)
        # print(x2.size())
        output1 = self.unfold(x1)
        output1 = torch.reshape(output1, (
        N, C//4, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))
        output2 = self.unfold(x2)
        # print(output2.size())
        output2 = torch.reshape(output2, (
            N, C // 4, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))
        output3 = self.unfold(x3)
        output3 = torch.reshape(output3, (
            N, C // 4, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))
        output4 = self.unfold(x4)
        output4 = torch.reshape(output4, (
            N, C // 4, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))

        x_se = torch.cat((output1, output2, output3, output4), dim=1)


        # SPC_out = input.view(N, self.S, C // self.S, H, W)  # bs,s,ci,h,w
        # for idx,conv in enumerate(self.convs):
        #     SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])
        # SE_out = torch.zeros_like(N,self.S, C // self.S,int(self.kernel_size[0] * self.kernel_size[1]),H*W)
        # for idx in range(4):
        #      A= self.unfold(SPC_out[:, idx, :, :, :])
        #      output = torch.reshape(A, (
        #      N, C//self.S, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))
        #      SE_out[:, idx, :, :, :]=output
        # output=SE_out.view(N, -1, 16,H*W)

        # output = self.unfold(input)  # (b, cxkxk, h*w)
        # # print(output.size())
        # output = torch.reshape(output, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
       

        # print(output.size())
        output = torch.mean(x_se, dim=2)
        # print(output.size())
        # output = self.fold(input)
        return output


class Extractor(nn.Module):
    r"""
    Build muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(self, backbone='vgg19',
                 cnn_layers=("relu1_1",),
                 upsample="nearest",
                 is_agg=False,
                 kernel_size=(4, 4),
                 stride=(4, 4),
                 dilation=1,
                 featmap_size=(112, 112),
                 device='cpu'):

        super(Extractor, self).__init__()
        self.device = torch.device(device)
        self.feature = SwinTransformer()
        # self.feature = SwinTransformer(img_size=224,
        #                                  patch_size=4,
        #                                  in_chans=3,
        #                                  num_classes=2,
        #                                  embed_dim=96,
        #                                  depths= [2, 2, 6, 2],
        #                                  num_heads=[3, 6, 12, 24],
        #                                  window_size=7,
        #                                  mlp_ratio=4,
        #                                  qkv_bias=True,
        #                                  qk_scale=False,
        #                                  drop_rate=0.0,
        #                                  drop_path_rate=0.1,
        #                                  ape=False,
        #                                  patch_norm=True,
        #                                  use_checkpoint=False)
        # self.feature=build_s()

        # self.feature = backbone_nets[backbone]()    # build backbone net
        self.feat_layers = cnn_layers
        self.is_agg = is_agg
        self.map_size = featmap_size
        self.upsample = upsample
        self.patch_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # feature processing
        padding_h = (self.patch_size[0] - self.stride[0]) // 2
        padding_w = (self.patch_size[1] - self.stride[1]) // 2
        self.padding = (padding_h, padding_w)
        self.replicationpad = nn.ReplicationPad2d((padding_w, padding_w, padding_h, padding_h))

        self.out_h = int((self.map_size[0] + 2*self.padding[0] - (self.dilation * (self.patch_size[0] - 1) + 1)) / self.stride[0] + 1)
        self.out_w = int((self.map_size[1] + 2*self.padding[1] - (self.dilation * (self.patch_size[1] - 1) + 1)) / self.stride[1] + 1)
        self.out_size = (self.out_h, self.out_w)
        # print(self.out_size)
        self.feat_agg_96 = AvgFeatAGG2d(channel=96, kernel_size=self.patch_size, output_size=self.out_size,
                                    dilation=self.dilation, stride=self.stride, device=self.device)
        self.feat_agg_192 = AvgFeatAGG2d(channel=192, kernel_size=self.patch_size, output_size=self.out_size,
                                     dilation=self.dilation, stride=self.stride, device=self.device)
        self.feat_agg_384 = AvgFeatAGG2d(channel=384, kernel_size=self.patch_size, output_size=self.out_size,
                                     dilation=self.dilation, stride=self.stride, device=self.device)
        self.feat_agg_768 = AvgFeatAGG2d(channel=768, kernel_size=self.patch_size, output_size=self.out_size,
                                     dilation=self.dilation, stride=self.stride, device=self.device)
        self.unfold = nn.Unfold(kernel_size=1, dilation=1, padding=0, stride=1)
        # self.features = torch.Tensor()


        # self.deconv3_1 = nn.ConvTranspose2d(in_channels=192,
        #                                     out_channels=192,
        #                                     kernel_size=3,
        #                                     stride=2,
        #                                     padding=1,
        #                                     output_padding=1,
        #                                     )

        # self.bn2 = nn.BatchNorm2d(num_features=192, )
        #
        # self.deconv4_1 = nn.ConvTranspose2d(in_channels=384,
        #                                     out_channels=384,
        #                                     kernel_size=3,
        #                                     stride=2,
        #                                     padding=1,
        #                                     output_padding=1,
        #                                     )
        # # self.deconv4_2 = nn.ConvTranspose2d(in_channels=384,
        # #                                     out_channels=384,
        # #                                     kernel_size=3,
        # #                                     stride=2,
        # #                                     padding=1,
        # #                                     output_padding=1,
        # #                                     )
        # self.bn3 = nn.BatchNorm2d(num_features=384, )
        #
        # self.deconv5_1 = nn.ConvTranspose2d(in_channels=768,
        #                                     out_channels=768,
        #                                     kernel_size=3,
        #                                     stride=2,
        #                                     padding=1,
        #                                     output_padding=1,
        #                                     )
        # # self.deconv5_2 = nn.ConvTranspose2d(in_channels=768,
        # #                                     out_channels=768,
        # #                                     kernel_size=3,
        # #                                     stride=2,
        # #                                     padding=1,
        # #                                     output_padding=1,
        # #                                     )
        # #
        # # self.deconv5_3 = nn.ConvTranspose2d(in_channels=768,
        # #                                     out_channels=768,
        # #                                     kernel_size=3,
        # #                                     stride=2,
        # #                                     padding=1,
        # #                                     output_padding=1,
        # #                                     )
        # #
        #
        # self.bn4 = nn.BatchNorm2d(num_features=768, )
        #
        # self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(192, 192, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(384, 384, kernel_size=3, stride=4, padding=1, dilation=1, output_padding=3)
        self.deconv3 = nn.ConvTranspose2d(768, 768, kernel_size=3, stride=8, padding=1, dilation=1, output_padding=7)

        self.f1 = BasicRFB_a(96, 96, stride=1, scale=1.0)
        self.f2 = BasicRFB_a(192, 192, stride=1, scale=1.0)
        self.f3 = BasicRFB_a(384, 384, stride=1, scale=1.0)
        self.f4 = BasicRFB_a(768, 768, stride=1, scale=1.0)

        # self.cur_conv_96 = nn.Sequential(ConvOffset2D(96),
        #                          nn.Conv2d(96, 96, 3, padding=1, stride=1),
        #                          nn.ReLU(inplace=True),
        #                          nn.BatchNorm2d(96),
        #                          )
        # self.cur_conv_192 = nn.Sequential(ConvOffset2D(192),
        #                                  nn.Conv2d(192, 192, 3, padding=1, stride=1),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.BatchNorm2d(192),
        #                                  )
        #
        # self.cur_conv_384 = nn.Sequential(ConvOffset2D(384),
        #                               nn.Conv2d(384, 384, 3, padding=1, stride=1),
        #                               nn.ReLU(inplace=True),
        #                               nn.BatchNorm2d(384),
        #                               )
        # self.cur_conv_768 = nn.Sequential(ConvOffset2D(768),
        #                                   nn.Conv2d(768, 768, 3, padding=1, stride=1),
        #                                   nn.ReLU(inplace=True),
        #                                   nn.BatchNorm2d(768),
        #                                   )

    def forward(self, input):
        feat_maps= self.feature(input)
        # print(feat_maps.size())
        x=feat_maps['stage3']
        features = torch.Tensor().to(self.device)
        # extracting features
        for _, feat_map in feat_maps.items():
            if self.is_agg:
                b,c,h,w=feat_map.size()
                # if h==56:
                #     feat_map = self.f1(feat_map)
                # if h == 28:
                #     feat_map = self.f2(feat_map)
                #     feat_map = self.deconv1(feat_map)
                # if h == 14:
                #     feat_map = self.f3(feat_map)
                #     feat_map = self.deconv2(feat_map)
                # if h == 7:
                #     feat_map = self.f4(feat_map)
                #     feat_map =self.deconv3(feat_map)

                # # resizing
                feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
                # print(feat_map.size())
                # B,c,H,W=feat_map.size()
                # if c==96:
                #     feat_map =self.cur_conv_96(feat_map)
                # if c == 192:
                #     feat_map = self.cur_conv_192(feat_map)
                # if c == 384:
                #     feat_map = self.cur_conv_384(feat_map)
                # if c == 768:
                #     feat_map = self.cur_conv_768(feat_map)


                # print(feat_map.size())
                feat_map = self.replicationpad(feat_map)
                # print(feat_map.size())
                # # aggregating features for every pixel
                if c==96:
                    feat_map =self.feat_agg_96(feat_map)
                if c== 192:
                    feat_map = self.feat_agg_192(feat_map)
                if c == 384:
                    feat_map = self.feat_agg_384(feat_map)
                if c == 768:
                    feat_map = self.feat_agg_768(feat_map)
                # feat_map = self.feat_agg(feat_map)
                # print(feat_map.size())
                # print(feat_map.size())
                # concatenating
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
                # print("Before reshaping:", features.shape)
            else:
                # print(1)
                # no aggregations
                # resizing
                feat_map = nn.functional.interpolate(feat_map, size=self.out_size, mode=self.upsample)
                # # print("feat_map:", feat_map.shape)
                # feat_map = self.replicationpad(feat_map)
                # # aggregating features for every pixel
                # feat_map = self.feat_agg(feat_map)
                # concatenating
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
                # print("Before reshaping:", features.shape)
        b, c, _= features.shape
        # print(self.out_size[0])
        features = torch.reshape(features, (b, c, self.out_size[0], self.out_size[1]))

        return features,x

    def feat_vec(self, input):
        # print('1')
        feat_maps = self.feature(input)
        features = torch.Tensor().to(self.device)
        # extracting features
        for name, feat_map in feat_maps.items():
            # resizing
            feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
            # print("feat_map:", feat_map.shape)
            # feat_map = feat_map[:,:,16:272,16:272]    # center crop 256x256
            # print("feat_map:", feat_map.shape)
            # padding (torch.nn.ReplicationPad2d(padding))
            feat_map = self.replicationpad(feat_map)
            b, c, h, w = feat_map.size()
            # print(feat_map.size())
            # # aggregating features for every pixel
            if c == 96:
                feat_map = self.feat_agg_96(feat_map)
            if c == 192:
                feat_map = self.feat_agg_192(feat_map)
            if c == 384:
                feat_map = self.feat_agg_384(feat_map)
            if c == 768:
                feat_map = self.feat_agg_768(feat_map)
            # aggregating features for every pixel
            # feat_map = self.feat_agg(feat_map)
            # print(feat_map.shape)
            # # unfolding
            # feat_map = self.unfold(feat_map)
            # print(feat_map.shape)
            # concatenating
            features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
        # print("Before reshaping:", features.shape)
        # reshaping features
        features = features.permute(0, 2, 1)  # (b, l, c)
        # print("After permute:", features.shape)
        features = torch.unbind(features, dim=0)  # [(l, c), ... (l, c)]; total b of (l, c)
        # print("Features len:", len(features))
        features = torch.cat(features, dim=0)  # (l*b, c); every (l, c) corresponds to a feature map
        return features

if __name__ == "__main__":
    import time
    vgg19_layers = ('relu1_1', 'relu1_2', 'relu2_1', 'relu2_2',
                    'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4')

    device = "cuda:1"
    extractor = Extractor(backbone="vgg19",
                          cnn_layers=vgg19_layers,
                          featmap_size=(256, 256),
                          device=device)

    time_s = time.time()
    extractor.to(device)
    batch_size = 1
    input = torch.Tensor(np.random.randn(batch_size, 3, 256, 256)).to(device)
    feats = extractor(input)

    print("Feature (n_samples, n_features):", feats.shape)
    print("Time cost:", (time.time() - time_s)/batch_size, "s")
