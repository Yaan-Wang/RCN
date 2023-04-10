import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from vgg19 import VGG19
from swinbackbone import SwinTransformer
# from rbfmodel import BasicRFB_a

# from swin_transformer import  build_s
# backbone nets
#backbone_nets = {'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19}
# backbone_nets = {'vgg19': VGG19}


# aggregation
class CR(nn.Module):
    """
    Aggregating features on feat maps: avg
    """

    def __init__(self, kernel_size, output_size=None, dilation=1, stride=1, device=torch.device('cpu'),p_r=3,p_b=0.9):
        super(CR, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, stride=stride)
        self.fold = nn.Fold(output_size=output_size, kernel_size=1, dilation=1, stride=1)
        self.output_size = output_size
        self.shift_size=p_r
        self.p_b=p_b
    # TODO: using unfold, fold, then xx.mean(dim=, keepdim=True)
    def forward(self, input):
        N, C, H, W = input.shape
        shifted_x = torch.roll(input, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        shifted_x_2 = self.unfold(shifted_x)  # (b, cxkxk, h*w)
        shifted_x_2 = torch.reshape(shifted_x_2, (
        N, C, int(self.kernel_size[0] * self.kernel_size[1]), int(self.output_size[0] * self.output_size[1])))
        shifted_x_2 = torch.roll(shifted_x_2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        shifted_x_2 = torch.mean(shifted_x_2, dim=2)

        output1 = self.unfold(input)  # (b, cxkxk, h*w)
        output1 = torch.reshape(output1, (N, C, int(self.kernel_size[0]*self.kernel_size[1]), int(self.output_size[0]*self.output_size[1])))
        output1 = torch.mean(output1, dim=2)
        output=output1+shifted_x_2*self.p_b
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
                 device='cpu',
                 p_k=0.7,
                 p_r=3,
                 p_b=0.9 ):

        super(Extractor, self).__init__()
        self.device = torch.device(device)
        self.feature =SwinTransformer(p_k=p_k)
        self.p_r=p_r
        self.p_b=p_b
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
        self.CR = CR(kernel_size=self.patch_size, output_size=self.out_size,
                                    dilation=self.dilation, stride=self.stride, device=self.device,p_r=self.p_r,p_b=self.p_b)
        self.unfold = nn.Unfold(kernel_size=1, dilation=1, padding=0, stride=1)

        # self.bn4 = nn.BatchNorm2d(num_features=768, )
        #
        # self.relu = nn.ReLU(inplace=True)
        self.deconv0 = nn.ConvTranspose2d(96, 96, kernel_size=1, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(192, 192, kernel_size=1, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(384, 384, kernel_size=1, stride=4, padding=1, dilation=1, output_padding=3)
        self.deconv3 = nn.ConvTranspose2d(768, 768, kernel_size=3, stride=8, padding=1, dilation=1, output_padding=7)
        #
        # self.f1 = BasicRFB_a(96, 96, stride=1, scale=1.0)
        # self.f2 = BasicRFB_a(192, 192, stride=1, scale=1.0)
        # self.f3 = BasicRFB_a(384, 384, stride=1, scale=1.0)
        # self.f4 = BasicRFB_a(768, 768, stride=1, scale=1.0)


    def forward(self, input):
        feat_maps= self.feature(input)
        # print(feat_maps.size())
        x=feat_maps['stage3']
        features = torch.Tensor().to(self.device)
        # extracting features
        for _, feat_map in feat_maps.items():
            if self.is_agg:
                # # resizing
                feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
                feat_map = self.replicationpad(feat_map)
                feat_map = self.CR(feat_map)
                # # concatenating
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
            else:
                # resizing
                feat_map = nn.functional.interpolate(feat_map, size=self.out_size, mode=self.upsample)
                # concatenating
                features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
        b, c, _= features.shape
        features = torch.reshape(features, (b, c, self.out_size[0], self.out_size[1]))
        return features,x

    def feat_vec(self, input):
        feat_maps = self.feature(input)
        features = torch.Tensor().to(self.device)
        # extracting features
        for name, feat_map in feat_maps.items():
            # resizing
            feat_map = nn.functional.interpolate(feat_map, size=self.map_size, mode=self.upsample, align_corners=True if self.upsample == 'bilinear' else None)
            feat_map = self.replicationpad(feat_map)
            # aggregating features for every pixel
            feat_map = self.CR(feat_map)
            # concatenating
            features = torch.cat([features, feat_map], dim=1)  # (b, ci + cj, h*w); (b, c, l)
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
