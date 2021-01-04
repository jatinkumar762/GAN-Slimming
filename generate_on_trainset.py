
import argparse, itertools, os, time
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import torch

from models.models import Generator, Discriminator
from utils.utils import *
from datasets.datasets import ImageDataset, PairedImageDataset


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0')
parser.add_argument('--cpus', default=4)
parser.add_argument('--batch_size', '-b', default=11, type=int, help='mini-batch size')
parser.add_argument('--task', type=str, default='A2B', choices=['A2B', 'B2A'])
parser.add_argument('--dataset', type=str, default='horse2zebra', choices=['summer2winter_yosemite', 'horse2zebra'])
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--rho', type=float, default=0, help='l1 loss weight')
parser.add_argument('--beta', type=float, default=20, help='GAN loss weight')
parser.add_argument('--lc', default='vgg', choices=['vgg', 'mse'], help='G content loss. vgg: perceptual; mse: mse')
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
args = parser.parse_args()

if args.task == 'A2B':
    source_str, target_str = 'A', 'B'
else:
    source_str, target_str = 'B', 'A'


result_dir = os.path.join('train_set_result',args.dataset,target_str)
create_dir(result_dir)

dataset_dir = os.path.join('./', 'datasets', args.dataset, 'train',source_str) 

transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ] # (0,1) -> (-1,1)

#test_ds = ImageFolder(dataset_dir, transforms_)
img_dataset = ImageDataset(dataset_dir, transforms_)
dataloader = DataLoader(img_dataset, batch_size=args.batch_size, num_workers=3, pin_memory=True, drop_last=True)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_source = Tensor(args.batch_size, args.input_nc, args.size, args.size)


## Networks
# G:
netG = Generator(args.input_nc, args.output_nc, quant=args.quant).cuda()

## Load pretrained model
# G:
netG.load_state_dict(torch.load('./pretrained_dense_model/horse2zebra/pth/netG_A2B_epoch_199.pth'))

# model in eval mode
netG.eval()

stats = ((0.5,0.5,0.5), (0.5,0.5,0.5))

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1).to(device='cuda')
    stds = torch.tensor(stds).reshape(1, 3, 1, 1).to(device='cuda')
    return images * stds + means

#All file names
file_name = dataloader.dataset.files

# genrate images
counter=0
for i, batch in enumerate(dataloader):
    input_img = Variable(input_source.copy_(batch))   
    output_images = netG(input_img)
    #print(i,len(batch), len(output_images),output_images[0].shape)
    output_images = denormalize(output_images,*stats)
    for image in output_images:
            #img_np = image.detach().cpu().numpy()
            #img_np = np.moveaxis(img_np, 0, -1)
            #img_np = (img_np + 1) / 2 # (-1,1) -> (0,1)
            #imsave(os.path.join(result_dir, 'image%d.png' % (counter)), img_as_ubyte(img_np))
            save_image(image, os.path.join(result_dir, '%s.png' % (file_name[counter].split('/')[-1])), normalize=False)
            #print(file_name[counter].split('/')[-1])
            counter+=1

torch.cuda.empty_cache()

