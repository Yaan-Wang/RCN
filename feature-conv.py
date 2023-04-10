import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from vgg19 import VGG19
from swinbackbone import SwinTransformer
from rbfmodel import BasicRFB_a

# from swin_transformer import  build_s
# backbone nets
#backbone_nets = {'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19}
backbone_nets = {'vgg19': VGG19}


# aggregation
class AvgFeatAGG2d(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, kernel_size, output_size=None, dilation=1, stride=1, device=torch.device('cpu')):
        super(AvgFeatAGG2d, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size

    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        output = self.unfold(input)  # (b, cxkxk, h*w)
        # print(  self.kernel_size)
        output = torch.reshape(output, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
        # print(output.shape)
        output = torch.mean(output, dim=2)
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
        self.feat_agg = AvgFeatAGG2d(kernel_size=self.patch_size, output_size=self.out_size,
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

        self.reducechanne0=  nn.Conv2d(96, 1, kernel_size=1)
        self.reducechanne1 = nn.Conv2d(192, 1, kernel_size=1)
        self.reducechanne2 = nn.Conv2d(384, 1, kernel_size=1)
        self.reducechanne3 = nn.Conv2d(768, 1, kernel_size=1)

    def forward(self, input):
        feat_maps= self.feature(input)
        # print(feat_maps.size())
        x3=feat_maps['stage3']
        x2 = feat_maps['stage2']
        x1 = feat_maps['stage1']
        x0 = feat_maps['stage0']
        if self.is_agg:
            feat_map_0 = nn.functional.interpolate(x0, size=self.map_size, mode=self.upsample,
                                                 align_corners=True if self.upsample == 'bilinear' else None)
            feat_map_0 = self.replicationpad(feat_map_0)
            feat_map_0_c=self.reducechanne0(feat_map_0)


            feat_map_1 = nn.functional.interpolate(x1, size=self.map_size, mode=self.upsample,
                                                   align_corners=True if self.upsample == 'bilinear' else None)
            feat_map_1 = self.replicationpad(feat_map_1)
            feat_map_1_c = self.reducechanne1(feat_map_1)


            feat_map_2 = nn.functional.interpolate(x2, size=self.map_size, mode=self.upsample,
                                                   align_corners=True if self.upsample == 'bilinear' else None)
            feat_map_2 = self.replicationpad(feat_map_2)
            feat_map_2_c = self.reducechanne2(feat_map_2)

            feat_map_3 = nn.functional.interpolate(x3, size=self.map_size, mode=self.upsample,
                                                   align_corners=True if self.upsample == 'bilinear' else None)
            feat_map_3 = self.replicationpad(feat_map_3)
            feat_map_3_c = self.reducechanne3(feat_map_3)

            feat_map_f_0=torch.cat([feat_map_0, feat_map_1_c,feat_map_2_c,feat_map_3_c], dim=1)
            feat_map_f_0 = self.feat_agg(feat_map_f_0)

            feat_map_f_1 = torch.cat([feat_map_1, feat_map_0_c, feat_map_2_c, feat_map_3_c], dim=1)
            feat_map_f_1 = self.feat_agg(feat_map_f_1)

            feat_map_f_2 = torch.cat([feat_map_2, feat_map_1_c, feat_map_0_c, feat_map_3_c], dim=1)
            feat_map_f_2 = self.feat_agg(feat_map_f_2)

            feat_map_f_3 = torch.cat([feat_map_3, feat_map_0_c, feat_map_1_c, feat_map_2_c], dim=1)
            feat_map_f_3 = self.feat_agg(feat_map_f_3)

            feat_map_f = torch.cat([feat_map_f_0, feat_map_f_1, feat_map_f_2, feat_map_f_3], dim=1)



        # features = torch.Tensor().to(self.device)
        # extracting features
        # for _, feat_map in feat_maps.items():
        #     if self.is_agg:
        #
        #         # b,c,h,w=feat_map.size()
        #         # if h==56:
        #         #     feat_map = self.f1(feat_map)
        #         # if h == 28:
        #         #     feat_map = self.f2(feat_map)
        #         #     feat_map = self.deconv1(feat_map)
        #         # if h == 14:
        #         #     feat_map = self.f3(feat_map)
        #         #     feat_map = self.deconv2(feat_map)
        #         # if h == 7:
        #         #     feat_map = self.f4(feat_map)
        #         #     feat_map =self.deconv3(feat_map)
        #
        #         # # resizing
        #         feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
        #         # print(feat_map.size())
        #         feat_map = self.replicationpad(feat_map)
        #         # print(feat_map.size())
        #         # # aggregating features for every pixel
        #         feat_map = self.feat_agg(feat_map)
        #         # print(feat_map.size())
        #         # print(feat_map.size())
        #         # concatenating
        #         features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
        #         # print("Before reshaping:", features.shape)
        #     else:
        #         # print(1)
        #         # no aggregations
        #         # resizing
        #         feat_map = nn.functional.interpolate(feat_map, size=self.out_size, mode=self.upsample)
        #         # # print("feat_map:", feat_map.shape)
        #         # feat_map = self.replicationpad(feat_map)
        #         # # aggregating features for every pixel
        #         # feat_map = self.feat_agg(feat_map)
        #         # concatenating
        #         features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
                # print("Before reshaping:", features.shape)
        b, c, _= feat_map_f.shape
        features = torch.reshape(feat_map_f, (b, c, self.out_size[0], self.out_size[1]))
        # print(features.size())
        return features,x3

    def feat_vec(self, input):
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
            # aggregating features for every pixel
            feat_map = self.feat_agg(feat_map)
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
