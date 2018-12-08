# Optical Flow network using RL and TVL1 energy function
# RL Approach for training flownet based network 

Author: Praveen Badimala Supervisor: Christian Bailer

We use FlowNets architecture as base network architecture
FlowNet is provided by Dosovitskiy et al.

We have used the components of flownet code implemented by ClementPinard
Ref: https://github.com/ClementPinard/FlowNetPytorch

we can train the network by any of the public datasets available like:
Flyingchairs
mpi-sintel
KITTI

Types:

There are two versions of rl networks
1) flow_reinforce_actor_critic_simple
   - this one uses two seperate networks, actor and critic network (ac_simple)
2) flow_reinforce_actor_critic
    - this one uses one single network, actor_critic_combined network (ac_single_net_a,ac_single_net_b)
flow_reinforce_actor_critic gave better results compared to the flow_reinforce_actor_critic_simple

Usage:

1)
Training flow_reinforce_actor_critic single network (ac_single_net_a and ac_single_net_b)
cmd:
 python flow_reinforce_actor_critic.py -b4 -j8 --epochs 20 --epoch-size 1200 --dataset flying_chairs flying_chairs_data_set_location

 -b4 mentions the batch size
 -j8 mentions the number of data loading threads
 --dataset can be changed to train on different datasets
   mpi_sintel dataset can be given as train dataset as --dataset mpi_sintel_clean
 -e can be used to evaluate
 --pretrained and --pretrained-aq-network can be used to load pretrained actor and actor_critic_combined network
	In evaluate mode, set of flow maps are generated for visualization

2)
Training flow_reinforce_actor_critic simple network
 python flow_reinforce_actor_critic_simple.py -b4 -j8 --epochs 20 --epoch-size 1200 --dataset flying_chairs flying_chairs_data_set_location
the command usage is same as flow_reinforce_actor_critic

Flow generation:
you can generate the flow using the run_inference class, you need to put the sample images in a data_folder and save the image
ending with 0.png,1.png . Any extensions can be used.

python run_inference.py --pretrained  actor_checkpoint.pth.tar --img-exts  png --data data_folder --output output_folder


dev environment and version:
	pytorch >= 0.4
	tensorboardX
	scipy
	argparse



	
