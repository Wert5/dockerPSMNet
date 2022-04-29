from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import copy
import imageio
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA
from PIL import Image

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savedisp', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
   from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
   from dataloader import KITTIloader2012 as ls

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img+test_left_img,all_right_img+test_right_img,all_left_disp+test_left_disp, False),
         batch_size= 1, shuffle= False, num_workers= 2, drop_last=False)

# TestImgLoader = torch.utils.data.DataLoader(
#          DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
#          batch_size= 1, shuffle= False, num_workers= 2, drop_last=False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

def loss(disp_true,pred_disp):
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp>0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 2)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
    torch.cuda.empty_cache()
    return 1-(float(torch.sum(correct))/float(len(index[0])))

def test(left, right, disp_L, imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output3 = model(imgL,imgR)

        pred_disp = output3.data.cpu()
        pred_disp = torch.squeeze(pred_disp, 1)

        print(disp_true.shape, pred_disp.shape)
        test_loss = loss(disp_true, pred_disp)
        print("CNN",test_loss)

        for i in range(len(left)):
            filename = os.path.join(args.savedisp, os.path.splitext(os.path.basename(left[i]))[0] + '.png')
            print("Saving",filename)
            img = torch.squeeze(pred_disp[i]).data.cpu().numpy()
            img = (img*256).astype('uint16')
            img = Image.fromarray(img)
            img.save(filename)

        return test_loss




def main():
    for batch_idx, (left,right,disp_L,imgL, imgR, disp_true) in enumerate(TestImgLoader):
        test(left,right,disp_L,imgL,imgR, disp_true)




if __name__ == '__main__':
    main()
