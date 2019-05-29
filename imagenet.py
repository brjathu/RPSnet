'''
RPS network training on imagenet dataset
Copyright (c) Jathushan Rajasegaran, 2019
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import pickle
import torch
import pdb
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import random


from rps_net import RPS_Net
from learner import Learner
from util import *
from cifar_dataset import CIFAR100


class args:
    epochs = 100
    checkpoint = "results/imagenet/RPSnet-IMAGENET_100_10_2k10"
    savepoint = ""
    data ='/raid/data/Machine_IIAI/imbalance/fastai_experiments/imagenet/ILSVRC/Data/CLS-LOC/'
    labels_data = "prepare/imagenet_100_10_2k.pkl"
    
    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 10
    dataset = "IMAGENET"
   
    L = 9
    N = 1
    lr = 0.001
    train_batch = 64
    test_batch = 64
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5


state = {key:value for key, value in args.__dict__.items() if not key.startswith('__') and not callable(key)}
print(state)

# Use CUDA
use_cuda = torch.cuda.is_available()

seed = random.randint(1, 10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)


def main():
    
    
    model = RPS_net(args).cuda() 
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        
    if not os.path.isdir("models/imagenet/"+args.checkpoint.split("/")[-1]):
        mkdir_p("models/imagenet/"+args.checkpoint.split("/")[-1])
    args.savepoint = "models/imagenet/"+args.checkpoint.split("/")[-1]




    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])    

    import torch.utils.data as data
    from PIL import Image 
    def default_loader(path):
	    return Image.open(path).convert('RGB')

    class ImageFilelist(data.Dataset):
        def __init__(self, root, flist,targets=None, transform=None, target_transform=None, loader=default_loader):
            self.root   = root
            self.imlist = flist		
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader
            self.targets=targets

        def __getitem__(self, index):
            impath = self.imlist[index]
            target = self.targets[index]
            img = self.loader(os.path.join(self.root,impath))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)           
            return img, target
        def __len__(self):
            return len(self.imlist)

    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    
    args.test_case = test_case

    a=pickle.load(open(args.labels_data,'rb'))

        
    for ses in range(start_sess, start_sess+1):

        ##############################  data loader for imagenet based upon file names #####################
        trn_fnames=a['trn_nms'][a[ses]['curent']]
        trn_labs=a['ytrain'][a[ses]['curent']]
        val_fnames=a['tst_nms'][a[ses]['test']]
        val_labs=a['ytest'][a[ses]['test']]

        if ses > 0:
            ex_fnames=a['trn_nms'][a[ses-1]['exmp']]
            ex_labs=a['ytrain'][a[ses-1]['exmp']]
            
            trn_fnames=np.concatenate((trn_fnames,np.tile(ex_fnames,1)))
            trn_labs=np.concatenate((trn_labs,np.tile(ex_labs,1)))

        train_dataset=ImageFilelist(root=args.data, flist=trn_fnames,targets=trn_labs,transform= transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset=ImageFilelist(root=args.data, flist=val_fnames,targets=val_labs,transform= transforms.Compose([
                transforms.Resize(230),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))


        train_sampler = None

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        testloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        ############################## data loader for imagenet based upon file names ######################


        
        if(ses==0):
            path = get_path(args.L,args.M,args.N)*0 
            path[:,0] = 1
            fixed_path = get_path(args.L,args.M,args.N)*0 
            train_path = path.copy()
            infer_path = path.copy()
        else:
            load_test_case = get_best_model(ses-1, args.checkpoint)
            if(ses%args.jump==0):   #get a new path
                fixed_path = np.load(args.checkpoint+"/fixed_path_"+str(ses-1)+"_"+str(load_test_case)+".npy")
                train_path = get_path(args.L,args.M,args.N)*0 
                path = get_path(args.L,args.M,args.N)
            else:
                if((ses//args.jump)*2==0):
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_test_case_x = get_best_model((ses//args.jump)*2-1, args.checkpoint)
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str((ses//args.jump)*2-1)+"_"+str(load_test_case_x)+".npy")
                path = np.load(args.checkpoint+"/path_"+str(ses-1)+"_"+str(load_test_case)+".npy")
            
                train_path = get_path(args.L,args.M,args.N)*0 
            infer_path = get_path(args.L,args.M,args.N)*0 
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        train_path[j,i]=1
                    if(fixed_path[j,i]==1 or path[j,i]==1):
                        infer_path[j,i]=1
            
        np.save(args.checkpoint+"/path_"+str(ses)+"_"+str(test_case)+".npy", path)
        
        
        if(ses==0):
            fixed_path_x = path.copy()
        else:
            fixed_path_x = fixed_path.copy()
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path_x[j,i]==0 and path[j,i]==1):
                        fixed_path_x[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path_x)
        
        
        
        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n",path)
        print("fixed_path\n",fixed_path)
        print("train_path\n", train_path)

        

        print(trn_fnames.shape)
        print(trn_labs.shape)
        print(val_fnames.shape)
        print(val_labs.shape)
        if(ses>0):
            print(ex_fnames.shape)
            print(ex_labs.shape)
        
        
        
        args.sess=ses      
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1)+'_'+str(load_test_case)+'_model_best.pth.tar')
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best['state_dict'])


        main_learner=Learner(model=model,args=args,trainloader=trainloader,
                             testloader=testloader,old_model=copy.deepcopy(model),
                             use_cuda=use_cuda, path=path, 
                             fixed_path=fixed_path, train_path=train_path, infer_path=infer_path)
        main_learner.learn()
        

        if(ses==0):
            fixed_path = path.copy()
        else:
            for j in range(args.L):
                for i in range(args.M):
                    if(fixed_path[j,i]==0 and path[j,i]==1):
                        fixed_path[j,i]=1
        np.save(args.checkpoint+"/fixed_path_"+str(ses)+"_"+str(test_case)+".npy", fixed_path)
        
        best_model = get_best_model(ses, args.checkpoint)
        
        
    print('done with session {:d}'.format(ses))
    print('#################################################################################')
    while(1):
        if(is_all_done(ses, args.epochs, args.checkpoint)):
            break
        else:
            time.sleep(10)
            
    
if __name__ == '__main__':
    main()

