import torch
import torch.nn.functional as F
from torch.autograd import Variable

'''
contains methods to wrap a second image to that of a first image using the flow vector
'''

# simple flow vector in pytorch vector form
def refine_flow_for_torch(flow,width,height):
    for i in range(0, height):
        for j in range(0, width):
            flow[i, j, 1] = (float(i)+flow[i,j,1]) / float(height - 1) * 2.0 - 1.0
            flow[i, j, 0] = (float(j)+flow[i,j,0])  / float(width - 1) * 2.0 - 1.0
    return flow

#  flow vector in pytorch vector form
def refine_flow_for_torch_optimized(flow,width,height):
    for i in range(0, height):
        for j in range(0, width):
            flow[i, j, 1] = (float(i)+flow[i,j,1]) / float(height - 1) * 2.0 - 1.0
            flow[i, j, 0] = (float(j)+flow[i,j,0])  / float(width - 1) * 2.0 - 1.0
    return flow

def refine_flow_for_torch_optimized_batch(flow,width,height):
    # the flow value shall be in the range of -1.0 to 1.0
    scale_factor = torch.FloatTensor([2.0]).cuda()
    h_cuda = scale_factor / torch.FloatTensor([height-1]).cuda()
    w_cuda = scale_factor / torch.FloatTensor([width-1]).cuda()
    for i in range(0, height):
        for j in range(0, width):
            flow[:,i, j, 1] = (float(i)+flow[:,i,j,1]) * h_cuda - 1.0
            flow[:,i, j, 0] = (float(j)+flow[:,i,j,0])  * w_cuda - 1.0
    return flow

# get the wrapped for a image and it's flow
def warp_image_torch_optimized(image,flow):
    height, width, channels = image.size()
    flow = refine_flow_for_torch_optimized(flow, width, height)
    flow = flow.cpu()

    torch_img = image.transpose(2, 1).transpose(1, 0).float()  # [1, 3, H, W]
    torch_img = torch_img.unsqueeze(0)

    flow = Variable(flow).unsqueeze(0)
    warped_img = F.grid_sample(torch_img, flow, 'bilinear')
    warped_img = warped_img.squeeze(0)
    warped_img = Variable(warped_img.data.transpose(0, 1).transpose(1, 2)).data
    return warped_img

# get the batch of wrapped image for a batch of images and their flow
def warp_image_torch_optimized_batch(image,flow):
    batch,height, width, channels = image.size()
    # convert the numpy flow to torch flow tensor
    flow = refine_flow_for_torch_optimized_batch(flow, width, height)
    flow = flow.cpu()
    torch_img = image.transpose(3, 2).transpose(2, 1).float()  # [1, 3, H, W]

    # find the wrapped image of second image using the flow
    warped_img = F.grid_sample(torch_img, Variable(flow).float(), 'bilinear')
    wrap_image =  warped_img.data.transpose(1, 2).transpose(2, 3)
    return wrap_image