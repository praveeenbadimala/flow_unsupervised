import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import flow_transforms
from scipy.ndimage import imread
from scipy.misc import imsave
from flow_reinforce_actor_critic import flow2rgb
from models import actor
import torch.nn as nns
from optflow import dual_tvl1_flow_generator
from optflow import compute_tvl1_energy
from datasets import listdataset
from multiscaleloss import EPE

'''
This file is used to generate the optical flow map of image samples from the network and visualize their output
along with their energies, the images need to be present in a folder with endings as 0.png,1.png or any
other file type and also EPE is computed, GT flow should be in same folder as images
with .flo extension
'''


parser = argparse.ArgumentParser(description='PyTorch FlowNet Actor network inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in current folder')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

# flying chairs trained network
parser.add_argument("--height", default=384, nargs='*', type=str, help="height of the input sample in trained net")
parser.add_argument("--width", default=512, nargs='*', type=str, help="weight of the input sample in trained net")

n_iter = 0

def main():
    global args, best_EPE, save_path
    args = parser.parse_args()
    data_dir = args.data
    required_h = int(args.height)
    required_w = int(args.width)

    img_exts = args.img_exts
    out_dir = ''
    if args.output  is not None:
        out_dir = args.output
        if not out_dir.endswith('/'):
            out_dir = out_dir+'/'


    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])

    img_pairs = []
    all_images = []
    for file in os.listdir(data_dir):
        if file.endswith('.'+img_exts[0]):
            all_images.append(file)

    all_flo = []
    for flo in os.listdir(data_dir):
        if flo.endswith('.flo'):
            all_flo.append(flo)

    for img in all_images:
        if img.endswith('0.'+img_exts[0]):
            check_image = img.replace('0.'+img_exts[0],'1.'+img_exts[0])
            if check_image in all_images:
                check_flow = img.replace('0.'+img_exts[0],'.flo')
                if check_flow in all_flo:
                    img_pairs.append([img,check_image,check_flow])


    print('{} samples found'.format(len(img_pairs)))

    # create model
    actor_data = actor.ActorLoad(args.pretrained).cuda()
    actor_model = torch.nn.DataParallel(actor_data).cuda()
    actor_model.eval()
    cudnn.benchmark = True

    for (img1_file, img2_file,target_flow) in img_pairs:

        img1_numpy = imread(os.path.join(data_dir, img1_file))
        img2_numpy = imread(os.path.join(data_dir, img2_file))
        target_flow = listdataset.load_flo(os.path.join(data_dir, target_flow))

        img1 = input_transform(img1_numpy)
        img2 = input_transform(img2_numpy)
        target_flow = target_transform(target_flow)

        input = torch.cat([img1, img2], 0).cuda()
        input_var = torch.autograd.Variable(input, volatile=True).unsqueeze(0)
        target_flow_var = torch.autograd.Variable(target_flow, volatile=True).unsqueeze(0).cuda()

        b, c, input_h, input_w = input_var.size()

        if input_h > required_h and input_w > required_w:
            input_var = nns.functional.adaptive_avg_pool2d(input_var, (required_h, required_w))
            img1_var = nns.functional.adaptive_avg_pool2d(torch.autograd.Variable(img1.unsqueeze(0), volatile=True),
                                                          (required_h, required_w))
            img2_var = nns.functional.adaptive_avg_pool2d(torch.autograd.Variable(img2.unsqueeze(0), volatile=True),
                                                          (required_h, required_w))
            target_flow_var = nns.functional.adaptive_avg_pool2d(target_flow_var,
                                                          (required_h, required_w))


        if input_h < required_h and input_w < required_w:
            input_var = nns.functional.upsample(input_var, size=(required_h, required_w), mode='bilinear')
            img1_var = nns.functional.upsample(torch.autograd.Variable(img1.unsqueeze(0), volatile=True),
                                           (required_h, required_w),mode='bilinear')
            img2_var = nns.functional.upsample(torch.autograd.Variable(img2.unsqueeze(0), volatile=True),
                                           size=(required_h, required_w),mode='bilinear')
            target_flow_var = nns.functional.upsample(target_flow_var,
                                               size=(required_h, required_w),mode='bilinear')

        if input_h == required_h and input_w == required_w:
            img1_var = torch.autograd.Variable(img1.unsqueeze(0), volatile=True)
            img2_var = torch.autograd.Variable(img2.unsqueeze(0), volatile=True)

        # compute output
        action, flow3, flow4, flow5, flow6 = actor_model(input_var)

        action_scaled = nns.functional.upsample(action, size=(required_h, required_w), mode='bilinear')
        action_max = action_scaled.max()
        action_min = action_scaled.min()
        max_value = torch.max(action_max.abs(), action_min.abs())
        imsave(out_dir+img1_file[:-4] + '_actor_flow.png',
               flow2rgb(action_scaled.clone().data[0].cpu().numpy(), max_value=max_value.cpu().data[0]))

        epe_error = EPE(action_scaled.cuda(), target_flow_var.cuda(), sparse=False, mean=True)

        target_flow_var = target_flow_var.clone()
        target_flow_var_max = target_flow_var.max()
        target_flow_var_min = target_flow_var.min()
        target_flow_var_max_value = torch.max(target_flow_var_max.abs(), target_flow_var_min.abs())

        imsave(out_dir+img1_file[:-4] + '_target_flow.png',
               flow2rgb(target_flow_var.data[0].cpu().numpy(), max_value=target_flow_var_max_value.data[0]))

        opencv_flow = dual_tvl1_flow_generator.compute_optical_flow_tvl1_opencv(img1_var.squeeze(0).data.numpy().transpose(1, 2, 0), img2_var.squeeze(0).data.numpy().transpose(1, 2, 0))
        opencv_flow_torch = torch.from_numpy(opencv_flow)
        opencv_flow_torch = opencv_flow_torch.transpose(1, 0).transpose(0, 2)
        opencv_flow_max = opencv_flow_torch.max()
        opencv_flow_min = opencv_flow_torch.min()
        opencv_flow_max_value = max(abs(opencv_flow_max), abs(opencv_flow_min))
        imsave(out_dir+img1_file[:-4] + '_opencv_tvl1_flow.png',
               flow2rgb(opencv_flow_torch.cpu().numpy(), max_value=opencv_flow_max_value))

        opencv_energy = compute_tvl1_energy.compute_tvl1_energy_optimized_batch(img1_var, img2_var,
                                                                                torch.autograd.Variable(
                                                                                    opencv_flow_torch.cuda().unsqueeze(
                                                                                        0)), img1_file[:-4])
        output_flow_energy = compute_tvl1_energy.compute_tvl1_energy_optimized_batch(img1_var, img2_var,
                                                                                     action_scaled[0].unsqueeze(0),
                                                                                     img1_file[:-4],True,out_dir)

        target_flow_energy = compute_tvl1_energy.compute_tvl1_energy_optimized_batch(img1_var, img2_var,
                                                                                     target_flow_var[0].unsqueeze(0),
                                                                                     img1_file[:-4])

        print(img1_file + ' opencv_tvl1_flow_energy:' + str(
            opencv_energy.data[0]) + ' flow_energy:' + str(output_flow_energy.data[0]) +
              ' target_flow_energy:'+str(target_flow_energy.data[0])+' epe error:'+str(epe_error.data[0]))

if __name__ == '__main__':
    main()
