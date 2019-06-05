#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:51:09 2019
@author: munawar
"""
import torchvision.transforms as transforms
import itertools
import operator
from functools import reduce
import random
import numpy as np
import torchvision

import pickle

d=pickle.load(open('meta_CIFAR100.pkl','rb'))
ytrain=d['ytr'];ytest=d['yt']

classes=np.arange(0,100)
inds_classes=[[np.where(ytrain==cl)] for cl in classes]


num_class = 100
task = 10
class_per_task = int(num_class/task)

def flatten_list(a): return np.concatenate(a).ravel()


def get_ind_sessions(ytrain,ytest):
    sessions=[]
    st=0;endd=class_per_task;
    for ii in range(task):
        sessions.append([np.arange(st,endd)])
        st=endd
        endd=st+class_per_task
    
    
    memb=2000
    
    
    indices_final=dict()
    
    
    for session in range(len(sessions)):
        ind_curr=flatten_list([np.where(ytrain==c) for c in flatten_list(sessions[session])])
        

        if session==0:
            ind_all=ind_curr
        else:
            ind_prev=indices_final[session-1]['exmp']
            ind_all=np.append(ind_curr,ind_prev)
            
        Ncl=1+sessions[session][0][-1]
        n_cl=int(memb/Ncl)
        ys=ytrain[np.array(ind_all)]
        exm=[]
        for ii in range(Ncl):
            ind_one_cl= ind_all[np.where(ys==ii)]
            random.shuffle(ind_one_cl)
            exm.append(ind_one_cl[:n_cl])
        exm=flatten_list(exm)
            
            
        
        all_test_classes=np.arange(0,class_per_task*(session+1)).astype(int)
        session_test_ind=flatten_list([np.where(ytest==c) for c in all_test_classes])
        
        indices_final[session]={'curent': ind_curr, 'exmp': exm,'test':session_test_ind}
    return indices_final

indices_final=get_ind_sessions(ytrain,ytest)


pickle.dump(indices_final, open('cifar100_'+str(task)+'.pkl','wb'))

 

    

