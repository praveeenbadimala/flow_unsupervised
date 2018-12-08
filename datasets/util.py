import numpy as np
import os

# source ref: https://github.com/ClementPinard/FlowNetPytorch

def split2list(images, split, default_split=0.9):

    np.random.shuffle(images)
    split_val = int(float(split) * len(images))
    train_images, test_images = images[:split_val], images[split_val:]

    if not os.path.exists(str('dataset_contents')):
        os.makedirs(str('dataset_contents'))

    with open('dataset_contents/test_images.txt', 'a') as the_file:
        for i,image_name in enumerate(test_images):
            the_file.write('no='+str(i)+' '+str(image_name)+'\n')

    with open('dataset_contents/train_images.txt', 'a') as the_file:
        for i,image_name in enumerate(train_images):
            the_file.write('no='+str(i)+' '+str(image_name)+'\n')


    return train_images,test_images