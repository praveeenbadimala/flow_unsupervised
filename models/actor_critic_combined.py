import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
import torch.nn as nns

# source ref: https://github.com/ClementPinard/FlowNetPytorch

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


class Actor_Critic_Combined(nn.Module):
    expansion = 1

    def __init__(self,actor_network,batchNorm=True):
        super(Actor_Critic_Combined,self).__init__()

        self.actor_network = actor_network

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   8,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 136,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 264,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 520,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 520, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        actor_flow2,actor_flow3,actor_flow4,actor_flow5,actor_flow6 = self.actor_network(x)
        b, _, height_vec, width_vec = x.size()
        action_scaled = nns.functional.upsample(actor_flow2, size=(height_vec, width_vec), mode='bilinear')

        critic_input = torch.cat([x.cuda(), action_scaled], 1)

        out_conv2 = self.conv2(self.conv1(critic_input))
        b, c, conv_h,conv_w = out_conv2.size()
        img_red = nns.functional.adaptive_avg_pool2d(x, (conv_h , conv_w))
        out_conv3 = self.conv3_1(self.conv3(torch.cat([actor_flow2,out_conv2,img_red],1)))
        b, c, conv_h, conv_w = out_conv3.size()
        img_red = nns.functional.adaptive_avg_pool2d(x, (conv_h, conv_w))
        out_conv4 = self.conv4_1(self.conv4(torch.cat([actor_flow3,out_conv3,img_red],1)))
        b, c, conv_h, conv_w = out_conv4.size()
        img_red = nns.functional.adaptive_avg_pool2d(x, (conv_h, conv_w))
        out_conv5 = self.conv5_1(self.conv5(torch.cat([actor_flow4,out_conv4,img_red],1)))
        b, c, conv_h, conv_w = out_conv5.size()
        img_red = nns.functional.adaptive_avg_pool2d(x, (conv_h, conv_w))
        out_conv6 = self.conv6_1(self.conv6(torch.cat([actor_flow5,out_conv5,img_red],1)))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # predict energy at each flow layer
        expected_energy_flow2 = flow2.sum(3).sum(2).sum(1)
        expected_energy_flow3 = flow3.sum(3).sum(2).sum(1)
        expected_energy_flow4 = flow4.sum(3).sum(2).sum(1)
        expected_energy_flow5 = flow5.sum(3).sum(2).sum(1)
        expected_energy_flow6 = flow6.sum(3).sum(2).sum(1)

        ret_val = {}
        ret_val['energy']=[expected_energy_flow2,expected_energy_flow3,expected_energy_flow4,expected_energy_flow5,expected_energy_flow6]
        ret_val['flow']=[actor_flow2,actor_flow3,actor_flow4,actor_flow5,actor_flow6]
        return ret_val

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def bias_parameters_critic(self):
        return [param for name, param in self.named_parameters() if 'bias' in name and 'actor_network' not in name]

    def weight_parameters_critic(self):
        return [param for name, param in self.named_parameters() if 'weight' in name and 'actor_network' not in name]


def Actor_Critic_combined_Load(actor_net,path=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        path : where to load pretrained network of actor critic single network. will create a new one if not set
    """
    model = Actor_Critic_Combined(actor_net,batchNorm=False)
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model