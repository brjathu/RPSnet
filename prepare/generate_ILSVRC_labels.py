#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 10:51:09 2019

@author: munawar
"""

import itertools
import operator
from functools import reduce
import random
import numpy as np
import torchvision
import pickle



d=pickle.load(open('meta_ILSVRC.pkl','rb'))

trn_nms=d['trn_names'];tst_nms=d['val_names']
ytrain=d['ytrain'];ytest=d['yval']


num_class = 100
task = 10
class_per_task = int(num_class/task)




top=np.where(np.array(ytrain)<num_class);ytrain=np.array(ytrain)[top];trn_nms=np.array(trn_nms)[top]
top=np.where(np.array(ytest)<num_class);ytest=np.array(ytest)[top];tst_nms=np.array(tst_nms)[top]


classes=np.arange(0,num_class)
random.shuffle(classes)
inds_classes=[[np.where(ytrain==cl)] for cl in classes]



def flatten_list(a): 
    return np.concatenate(a).ravel()


def get_ind_sessions(ytrain,ytest):
    sessions=[]
    st=0;endd=class_per_task;
    for ii in range(task):
        sessions.append([np.arange(st,endd)])
        st=endd
        endd=st+class_per_task
    
    
    memb=20000
    
    
    indices_final=dict()
    
    
    for session in range(len(sessions)):

        ind_curr=flatten_list([np.where(ytrain==c)[0] for c in flatten_list(sessions[session])])
        

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
        session_test_ind=flatten_list([np.where(ytest==c)[0] for c in all_test_classes])
        
        indices_final[session]={'curent': ind_curr, 'exmp': exm,'test':session_test_ind}
    return indices_final

indices_final=get_ind_sessions(ytrain.flatten(),ytest.flatten())

indices_final['trn_nms']=trn_nms
indices_final['tst_nms']=tst_nms
indices_final['ytrain']=ytrain
indices_final['ytest']=ytest

pickle.dump(indices_final, open('imagenet_'+str(num_class)+'_'+str(task)+'_2k.pkl','wb'))

 
