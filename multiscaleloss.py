import torch
import torch.nn as nn
import optflow.compute_tvl1_energy as compute_tvl1_energy

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)

    if sparse:
        EPE_map = EPE_map[target_flow != 0]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()


def multiscale_energy_loss(network_output_energy, target_flow,img1,img2, weights=None, sparse=False):
    def one_scale_mod(output, target, sparse,img1,img2):
        b, _, h, w = target.size()
        down_sample_img1 =nn.functional.adaptive_avg_pool2d(img1, (h, w))
        down_sample_img2 = nn.functional.adaptive_avg_pool2d(img2, (h, w))
        target_energy =  compute_tvl1_energy.compute_tvl1_energy_optimized_batch(down_sample_img1,
                                                                down_sample_img2,
                                                                target)
        l1_loss = (output - target_energy).abs().sum() / target_energy.size(0)
        return l1_loss

    if type(network_output_energy) not in [tuple, list]:
        network_output_energy = [network_output_energy]
    if weights is None:
        weights = [0.46,0.23,0.23,0.46]  # more preference for starting layers
    assert(len(weights) == len(network_output_energy))

    loss = 0
    flow_index = 0
    for output, weight in zip(network_output_energy, weights):
        loss += weight * one_scale_mod(output, target_flow[flow_index], sparse,img1,img2)
        flow_index = flow_index + 1
    return loss

def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.upsample(output, size=(h,w), mode='bilinear')
    return EPE(upsampled_output, target, sparse, mean=True)