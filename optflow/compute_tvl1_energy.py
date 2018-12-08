import numpy as np
from . import torch_warp as t_warp
import torch
from torch.autograd import Variable
import scipy

# this file is to find the TVL1 energy of the optical flow vector


def compute_flow_gradient(flowvector,pixelposx,pixelposy,imgwidth,imgheight):
    ux_grad = 0
    uy_grad = 0
    if pixelposx > 0 and pixelposx < (imgwidth-1):
        ux_prev = flowvector[pixelposy,pixelposx-1][0]
        ux_next = flowvector[pixelposy,pixelposx+1][0]
        ux_grad = float(ux_next - ux_prev)/2.0
    if pixelposy > 0 and pixelposy < (imgheight-1):
        uy_prev = flowvector[pixelposy-1,pixelposx][1]
        uy_next = flowvector[pixelposy+1,pixelposx][1]
        uy_grad = float(uy_next - uy_prev)/2.0
    return [ux_grad,uy_grad]

def compute_intensity_gradient(img_channel,pixel_x,pixel_y,img_width,img_height):
    ix_grad = 0
    iy_grad = 0
    if pixel_x >= 0 and pixel_x < (img_width-1):
        ix_next = img_channel[pixel_y][pixel_x+1]
        ix_current = img_channel[pixel_y][pixel_x]
        ix_grad = ix_next - ix_current
    if pixel_y >= 0 and pixel_y < (img_height-1):
        iy_next = img_channel[pixel_y+1][pixel_x]
        iy_current = img_channel[pixel_y][pixel_x]
        iy_grad = iy_next - iy_current
    return (ix_grad , iy_grad)

def compute_flow_gradient_optimized(flowvector,pixelposx,pixelposy,imgwidth,imgheight):
    ux_grad = 0.0
    uy_grad = 0.0
    if pixelposx > 0 and pixelposx < (imgwidth-1):
        ux_grad = (flowvector[pixelposy,pixelposx+1][0] - flowvector[pixelposy,pixelposx-1][0])/2.0
    if pixelposy > 0 and pixelposy < (imgheight-1):
        uy_grad = (flowvector[pixelposy+1,pixelposx][1] - flowvector[pixelposy-1,pixelposx][1]) /2.0
    return torch.Tensor([ux_grad,uy_grad])

# tvl1 energy of single image
def compute_tvl1_energy_optimized(img1,img2,flow):
    img1 = img1.transpose(0,1).transpose(1,2)
    img2 = img2.transpose(0,1).transpose(1,2)
    flow = flow.transpose(0,1).transpose(1,2)
    height, width, no_of_chans = img1.size()
    wrapped_first_image  = t_warp.warp_image_torch_optimized(img2,flow.data.clone())

    grad_vec = torch.abs(wrapped_first_image - img1.data)
    grad_vec = torch.norm(grad_vec,2,2)
    imag_grad = grad_vec.sum()

    ux_grad = (flow[:,2:,0]-flow[:,:width-2,0])/2.0
    uy_grad = (flow[2:,:,1]-flow[:height-2,:,1])/2.0
    ux_grad = ux_grad * ux_grad
    uy_grad = uy_grad * uy_grad
    sum = (ux_grad[1:-1, :] + uy_grad[:, 1:-1]).pow(0.5)
    grad_loss = sum.sum() + ux_grad[0, :].sum() + ux_grad[height - 1, :].sum() + uy_grad[:, 0].sum() + uy_grad[:,
                                                                                                 width - 1].sum()
    energy = grad_loss+imag_grad
    return energy

# This function is one which is used to compute the tvl1 energy associated with flow in batches
def compute_tvl1_energy_optimized_batch(img1,img2,flow,image_name='test',doTest=False,test_folder=''):

    img1 = img1.transpose(1,2).transpose(2,3)
    img2 = img2.transpose(1,2).transpose(2,3)

    flow = flow.transpose(1,2).transpose(2,3)
    batch,height, width, no_of_chans = img1.size()

    # get the wrapped image
    wrapped_first_image  = t_warp.warp_image_torch_optimized_batch(img2,flow.data.clone())

    # during inference phase write the wrapped and original image in a directory
    if doTest==True:
        scipy.misc.imsave(image_name +'_test_0.jpg', img1[0].data.numpy()+ np.array([0.411,0.432,0.45]))
        scipy.misc.imsave(image_name +'_test_1.jpg', img2[0].data.numpy()+ np.array([0.411,0.432,0.45]))
        scipy.misc.imsave(image_name +'_test_w.jpg', wrapped_first_image[0].numpy()+ np.array([0.411,0.432,0.45]))


    # find the image intensity values between the wrapped and first image
    grad_vec = torch.abs(wrapped_first_image - img1.data) * 255 * 0.15

    # constant penalty of value '23' for the black pixels in all the channels
    for i in range(batch):
        for j in range(no_of_chans):
            grad_vec[i,:,:,j][torch.eq(wrapped_first_image.sum(3),0)[i]] = 23

    grad_vec = torch.norm(grad_vec,2,3)
    # data fidelity term scalar for each batch sample
    imag_grad = grad_vec.sum(2).sum(1)

    # compute the flow smoothness
    # compute the sum of ux_dx and ux_dy values from 1st to n-1 values for flow in x direction
    ux_grad_x = (flow[:,:,1:,0] - flow[:,:,:width-1,0])
    ux_grad_y = (flow[:,1:,:,0] - flow[:,:height-1,:,0])


    ux_grad_x_mx = ux_grad_x * ux_grad_x
    ux_grad_y_my = ux_grad_y * ux_grad_y

    # compute the mod of ux_dx and ux_dy
    # sum (1..n-1,1...n-1)
    sum = (ux_grad_x_mx[:, :height - 1, :width - 1] + ux_grad_y_my[:, :height - 1, :width - 1]).pow(0.5)
    sum_ux = sum.sum(2).sum(1) + ux_grad_x[:, height - 1, :].sum(1).float() + ux_grad_y[:, :, width - 1].sum(1).float()

    # compute the sum of uy_dx and uy_dy values from 1st to n-1 values for flow in y direction
    uy_grad_x = (flow[:,:,1:,1] - flow[:,:,:width-1,1])
    uy_grad_y = (flow[:,1:,:,1] - flow[:,:height-1,:,1])
    uy_grad_x_mx = uy_grad_x * uy_grad_x
    uy_grad_y_my = uy_grad_y * uy_grad_y
    sum = (uy_grad_x_mx[:, :height - 1, :width - 1] + uy_grad_y_my[:, :height - 1, :width - 1]).pow(0.5)
    sum_uy = sum.sum(2).sum(1) + uy_grad_x[:, height - 1, :].sum(1).float() + uy_grad_y[:, :, width - 1].sum(1).float()
    grad_loss = sum_ux + sum_uy

    # total energy is sum of appearance constancy and flow smoothness
    energy = grad_loss+Variable(imag_grad.cuda())
    return energy

