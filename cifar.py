'''
RPS network training on CIFAR100
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


from rps_net import RPS_net_cifar
from learner import Learner
from util import *
from cifar_dataset import CIFAR100


class args:

    checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
    labels_data = "prepare/cifar100_10.pkl"
    savepoint = ""
    
    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 2.5
    dataset = "CIFAR"
    
    epochs = 100
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
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


    model = RPS_net_cifar(args).cuda() 
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    
    
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
        
    if not os.path.isdir("models/CIFAR100/"+args.checkpoint.split("/")[-1]):
        mkdir_p("models/CIFAR100/"+args.checkpoint.split("/")[-1])
    args.savepoint = "models/CIFAR100/"+args.checkpoint.split("/")[-1]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = CIFAR100
    
    
    
    
    start_sess = int(sys.argv[2])
    test_case = sys.argv[1]
    args.test_case = test_case

    inds_all_sessions=pickle.load(open(args.labels_data,'rb'))

        
    for ses in range(start_sess, start_sess+1):
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
                path = get_path(args.L,args.M,args.N)
                train_path = get_path(args.L,args.M,args.N)*0 
            else:
                if((ses//args.jump)==0):
                    fixed_path = get_path(args.L,args.M,args.N)*0
                else:
                    load_test_case_x = get_best_model((ses//args.jump)*args.jump-1, args.checkpoint)
                    fixed_path = np.load(args.checkpoint+"/fixed_path_"+str((ses//args.jump)*args.jump-1)+"_"+str(load_test_case_x)+".npy")
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
        
        
        print('Starting with session {:d}'.format(ses))
        print('test case : ' + str(test_case))
        print('#################################################################################')
        print("path\n",path)
        print("fixed_path\n",fixed_path)
        print("train_path\n", train_path)
        
    
        ind_this_session=inds_all_sessions[ses]    
        ind_trn= ind_this_session['curent']
        if ses > 0: ind_trn = np.concatenate([ind_trn,  np.tile(inds_all_sessions[ses-1]['exmp'],int(1))]).ravel()
        ind_tst=inds_all_sessions[ses]['test']

        
        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train,ind=ind_trn)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True,num_workers=args.workers)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test,ind=ind_tst)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        
        
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
    
        cfmat = main_learner.get_confusion_matrix(infer_path)
        np.save(args.checkpoint+"/confusion_matrix_"+str(ses)+"_"+str(test_case)+".npy", cfmat)
        
        
    print('done with session {:d}'.format(ses))
    print('#################################################################################')
    while(1):
        if(is_all_done(ses, args.epochs, args.checkpoint)):
            break
        else:
            time.sleep(10)
            
            
            
if __name__ == '__main__':
    main()
