import argparse
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import shutil
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
import models
import datasets
from multiscaleloss import realEPE, EPE, multiscale_energy_loss
import datetime
from tensorboardX import SummaryWriter
import numpy as np
from optflow import compute_tvl1_energy
import lr_scheduler
from models import actor
from models import actor_critic_combined
import torch.nn as nns
import scipy
from optflow import dual_tvl1_flow_generator 

# ref: https://github.com/ClementPinard/FlowNetPytorch

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

dataset_names = sorted(name for name in datasets.__all__)


parser = argparse.ArgumentParser(description='PyTorch FlowNet Actor net Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
parser.add_argument('-s', '--split', default=0.8,
                    help='test-val split file')
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1200, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model of actor network')
parser.add_argument('--pretrained-aq-network', dest='pretrainedaq', default=None,
                    help='path to pre-trained model of q network')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder')
parser.add_argument('--milestones', default=[5,8,12], nargs='*', help='epochs ai which learning rate is divided by 2')

best_valid_loss = -1
best_train_loss = -1
n_iter = 0


def main():
    global args, best_valid_loss, best_train_loss, save_path
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        'AC_flownets',
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join(args.dataset,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    target_transform = transforms.Compose([
        flow_transforms.ArrayToTensor()
    ])

    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = datasets.__dict__[args.dataset](
        args.data,
        transform=input_transform,
        target_transform=target_transform,
        co_transform=None,
        split=args.split
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    actor_net_str = 'AC_flownets_actor'
    actor_critic_combined_str = 'AC_flownets_actor_critic'

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(actor_net_str))
    else:
        print("=> creating model '{}'".format(actor_net_str))

    if args.pretrainedaq:
        print("=> using pre-trained model '{}'".format(actor_critic_combined_str))
    else:
        print("=> creating model '{}'".format(actor_critic_combined_str))

    actor_data = actor.ActorLoad(args.pretrained).cuda()
    actor_model = torch.nn.DataParallel(actor_data).cuda()
    cudnn.benchmark = True


    actor_critic_combined_data = load_actor_critic_combined_network(actor_model,args.pretrainedaq).cuda()
    actor_critic_combined_model = torch.nn.DataParallel(actor_critic_combined_data).cuda()

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))

    #get parameters of actor network
    actor_weight_params = actor_model.module.weight_parameters()
    actor_bias_params = actor_model.module.bias_parameters()

    # get only parameters of critic network part and dont include actor network parameters
    critic_weight_params = actor_critic_combined_model.module.weight_parameters_critic()
    critic_bias_params = actor_critic_combined_model.module.bias_parameters_critic()

    param_groups = [{'params': actor_bias_params, 'weight_decay': args.bias_decay},
                    {'params': actor_weight_params, 'weight_decay': args.weight_decay}]

    param_groups_q_net =  [{'params':critic_weight_params , 'weight_decay': args.bias_decay},
                    {'params': critic_bias_params, 'weight_decay': args.weight_decay}]


    if args.solver == 'adam':
        actor_optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(args.momentum, args.beta))
        critic_optimizer = torch.optim.Adam(param_groups_q_net, args.lr,
                                           betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        actor_optimizer = torch.optim.SGD(param_groups, args.lr,
                                    momentum=args.momentum)
        critic_optimizer = torch.optim.SGD(param_groups_q_net, args.lr,
                                          momentum=args.momentum)

    if args.evaluate:
        best_valid_loss = validate(val_loader, actor_model, 0)
        return

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optimizer, milestones=args.milestones, gamma=0.5)
    critic_scheduler = lr_scheduler.MultiStepLR(critic_optimizer, milestones=args.milestones, gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        actor_scheduler.step()
        critic_scheduler.step()

        # train for one epoch
        train_loss = train_policy_q_network(train_loader,actor_model,actor_critic_combined_model, actor_optimizer,critic_optimizer, epoch, train_writer)
        train_writer.add_scalar('loss computed', train_loss, epoch)

        # evaluate on validation set
        valid_loss = validate(val_loader, actor_model, epoch)
        test_writer.add_scalar('valid loss ', valid_loss, epoch)

        isBest = False

        if best_valid_loss < 0:
            best_valid_loss = valid_loss

        if valid_loss <= best_valid_loss:
            isBest = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': actor_net_str,
            'state_dict': actor_model.module.state_dict(),
            'best_EPE': best_valid_loss,
        }, isBest, filename='actor_checkpoint.pth.tar')

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': actor_net_str,
            'state_dict': actor_critic_combined_model.module.state_dict(),
            'best_EPE': best_valid_loss,
        }, isBest, filename='combine_model_checkpoint.pth.tar',best_model='combine_model_best.pth.tar')

def load_actor_critic_combined_network(actor_net,path=None):
    return actor_critic_combined.Actor_Critic_combined_Load(actor_net,path)

def train_policy_q_network(train_loader,actor_model,actor_critic_combined_model, actor_optimizer,critic_optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epes = AverageMeter()
    energy_vals = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    actor_model.cuda()
    actor_critic_combined_model.cuda()

    end = time.time()

    actor_critic_combined_model.train()
    actor_model.train()

    for i, (input, flow) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = flow.cuda(async=True)
        input2 = [j.cuda() for j in input]
        input_var = torch.autograd.Variable(torch.cat(input2, 1))

        actor_model.zero_grad()
        actor_critic_combined_model.zero_grad()

        # get energy and flow values from the single network
        ret_val = actor_critic_combined_model(input_var)

        action_vals = ret_val['flow']
        energy_vals_scale = ret_val['energy']

        b,_,height_vec,width_vec = input_var.size()
        action_scaled = nns.functional.upsample(action_vals[0], size=(height_vec, width_vec), mode='bilinear')
        q_net_target_energy = compute_tvl1_energy.compute_tvl1_energy_optimized_batch(torch.autograd.Variable(input[0]),
                                                                                      torch.autograd.Variable(input[1]),
                                                                                      action_scaled)

        #l1 loss for energy computed in last layer of critic network
        l1_loss = (energy_vals_scale[0] - q_net_target_energy).abs().sum() / q_net_target_energy.size(0)

        '''
        commenting the following code line makes the network architecture as ac_single_net a else its ac_single_net_b
        '''
        l1_loss = l1_loss + multiscale_energy_loss(energy_vals_scale[1:],action_vals[1:],torch.autograd.Variable(input[0]),torch.autograd.Variable(input[1]))

        # compute gradient and do optimization step
        # backpropagate and update only the critic network weights
        critic_optimizer.zero_grad()
        l1_loss.backward()
        critic_optimizer.step()

        # make gradients zero for both actor and critic network part
        actor_critic_combined_model.zero_grad()

        # forward pass again and predict the energy
        ret_val = actor_critic_combined_model(input_var)
        energy_vals_scale = ret_val['energy']

        # make gradients zero for both actor and critic network part
        actor_critic_combined_model.zero_grad()
        actor_optimizer.zero_grad()

        # backpropagate the energy as loss
        energy_vals_scale[0].backward(torch.ones(energy_vals_scale[0].size(0)).cuda())

        #find epe error with ground truth
        epe_error = EPE(action_scaled.cuda(), torch.autograd.Variable(target), sparse=False, mean=True)
        epes.update(epe_error.data[0], energy_vals_scale[0].size(0))

        train_writer.add_scalar('flow_epe', epe_error.data[0], n_iter)
        train_writer.add_scalar('q_train_loss', l1_loss.data[0], n_iter)
        train_writer.add_scalar('target_energy',q_net_target_energy.data[0],n_iter)
        train_writer.add_scalar('q_energy',energy_vals_scale[0].data[0],n_iter)
        energy_vals.update(energy_vals_scale[0].data[0], 1)

        # update the actor network part weights
        actor_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t EPE {5}\t Energy {6} \t l1_loss {7}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, epe_error.data[0],energy_vals, l1_loss.data[0]))

        n_iter += 1
        if i >= epoch_size:
            break

    return epes.avg

def validate(val_loader, actor_model, epoch):
    global args

    if not os.path.exists(str(epoch)):
        os.makedirs(str(epoch))

    batch_time = AverageMeter()
    validation_epe = AverageMeter()

    # switch to evaluate mode
    actor_model.eval()

    end = time.time()
    for i, (input, flow) in enumerate(val_loader):
        input2 = torch.cat(input, 1).cuda()
        flow = flow.cuda(async=True)

        # compute different flow
        action_list = actor_model(torch.autograd.Variable(input2))
        action = action_list[0]
        epe_error = realEPE(action, torch.autograd.Variable(flow), sparse=False)
        b, _, h, w = flow.size()

        # record EPE
        validation_epe.update(epe_error.data[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        action_scaled = nns.functional.upsample(action, size=(h, w), mode='bilinear')
        max_val = action_scaled.max()
        min_val = action_scaled.min()
        max_value = torch.max(max_val.abs(), min_val.abs())

        if i<10:
            # store image_0,image_1, flow map of ground truth,flow map of the network,flow map of opencv
            scipy.misc.imsave(str(epoch)+'/image' + str(i) + '0.jpg', input[0][0].numpy().transpose(1, 2, 0)+ np.array([0.411,0.432,0.45]))
            scipy.misc.imsave(str(epoch)+'/image' + str(i) + '1.jpg',
                              input[1][0].numpy().transpose(1, 2, 0)+ np.array([0.411,0.432,0.45]))
            scipy.misc.imsave(str(epoch)+'/flow_gt'+str(i)+'.jpg',flow2rgb(flow[0].cpu().numpy(), max_value=max_value.cpu().data[0]))
            scipy.misc.imsave(str(epoch)+'/flow_actor' + str(i) + '.jpg',flow2rgb(action_scaled.data[0].cpu().numpy(), max_value=max_value.cpu().data[0]))

            opencv_flow = dual_tvl1_flow_generator.compute_optical_flow_tvl1_opencv(
                input[0][0].numpy().transpose(1, 2, 0),
                input[1][0].numpy().transpose(1, 2, 0))
            opencv_flow_torch = torch.from_numpy(opencv_flow)
            opencv_flow_torch = opencv_flow_torch.transpose(1, 0).transpose(0, 2)
            opencv_flow_max = opencv_flow_torch.max()
            opencv_flow_min = opencv_flow_torch.min()
            opencv_flow_max_value = max(abs(opencv_flow_max), abs(opencv_flow_min))
            scipy.misc.imsave(str(epoch)+'/opencv_flow' + str(i) + '.jpg',
                   flow2rgb(opencv_flow_torch.cpu().numpy(), max_value=opencv_flow_max_value))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t validation {3}'
                  .format(i, len(val_loader), batch_time, epe_error.data[0]))

    print(' * validation loss {:.3f}'.format(validation_epe.avg))

    return validation_epe.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',best_model='actor_model_best.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,best_model))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    normalized_flow_map = flow_map/max_value
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

if __name__ == '__main__':
    main()

