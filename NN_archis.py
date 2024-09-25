#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:03:45 2024

@author: mu273311

adding m<<1 to total so that prob is never completely 0
"""

import torch
import torch.nn as nn
#import torch.optim as optim
import torch.distributions as dist
#from matplotlib import pyplot as plt
import numpy as np


m=.00001



#%%thresh_hat_med_block
"""    
thresh_hat_med_block_BNN puts a "soft" hat on sp src, forcing it to be under 5% of bkg. also bkg is blocked to be less than median

"""    
class NN_archi(nn.Module):#,sp_hat: bool = False, med_block=False):
    def __init__(self,obs=None,sp_hat = False, med_block: bool =False):
        super(NN_archi, self).__init__()
        
        self.sp_hat=sp_hat
        self.med_block=med_block
        self.obs=torch.tensor(obs)
        
        output_dim=1
        en_input_dim = 1
        sp_input_dim = 2
        sgn_sp_hd=10 #hidden dimension for signal in space - works ok with 10
        bkg_sp_hd=10 #hidden dimension for background in space
        sgn_en_hd=5 #hidden dimension for signal in energy
        bkg_en_hd=5 #hidden dimension for background in energy
        
        #self.bkg_sp=torch.nn.Parameter(torch.tensor(target_sp_mean), requires_grad=True) #remove if fail
        self.sgn_sp = nn.Sequential(
            nn.Linear(sp_input_dim,sgn_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_sp_hd,sgn_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_sp_hd,sgn_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_sp_hd,sgn_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_sp_hd,output_dim),
            nn.ELU())
        
        self.bkg_sp = nn.Sequential(
            nn.Linear(sp_input_dim,bkg_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_sp_hd,bkg_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_sp_hd,bkg_sp_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_sp_hd,output_dim),
            nn.ELU())
        
        self.sgn_en = nn.Sequential(
            nn.Linear(en_input_dim,sgn_en_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_en_hd,sgn_en_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_en_hd,sgn_en_hd),
            nn.LeakyReLU(),
            nn.Linear(sgn_en_hd,output_dim),
            nn.ELU())

        
        self.bkg_en = nn.Sequential(
            nn.Linear(en_input_dim,bkg_en_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_en_hd,bkg_en_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_en_hd,bkg_en_hd),
            nn.LeakyReLU(),
            nn.Linear(bkg_en_hd,output_dim),
            nn.ELU())

        
        
        self.sgn_sp.apply(self.initialize_weights)
        self.sgn_en.apply(self.initialize_weights)
        self.bkg_sp.apply(self.initialize_weights)
        self.bkg_en.apply(self.initialize_weights)
       

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def standardize_inputs(self): #expect normal np arrays (shapes (l) & (n,n,2))
        obs=self.obs
        l=min(np.shape(obs))
        n=max(np.shape(obs))
        s_scale_norm=(np.mgrid[0:1:n*1j, 0:1:n*1j]).transpose()
        e_scale_norm=np.linspace(0,1,l)
        input_sp=torch.tensor(np.float32(s_scale_norm))#change to BNN format
        input_en=torch.tensor(np.float32(np.reshape(e_scale_norm,(-1,1))))#change to BNN format
        return input_en, input_sp
            
    def space_hat(self): #hat function
        input_en, input_sp=self.standardize_inputs()
        n=len(input_sp)
        space_hat=np.array([[[(i-(n-1)/2)**2+(j-(n-1)/2)**2] for i in range(n)]for j in range(n)])
        space_hat=torch.tensor(np.where(space_hat>((n-1)/3)**2,0,1))
        return space_hat
        

    def final_outputs(self): #outputs pytorch tensors by default but can ask for np arrays instead
        input_en, input_sp=self.standardize_inputs()
        sp_bkg=self.bkg_sp(input_sp)+1
        if self.med_block==True:
            obs_med=torch.mean(torch.sum(self.obs,axis=-1).float())
            sp_bkg=torch.where(sp_bkg<obs_med,sp_bkg,obs_med)
        sp_sgn=self.sgn_sp(input_sp)+1
        if self.sp_hat==True:
            sp_sgn=torch.where((self.space_hat()==1),sp_sgn,0)
        elif self.sp_hat=='soft':
            sp_sgn=torch.where((self.space_hat()==1)|( sp_sgn<sp_bkg*.05),sp_sgn,sp_bkg*.05)
        en_sgn=self.sgn_en(input_en)+1
            
        if torch.sum(en_sgn)==0:
            en_sgn=1-en_sgn
        en_sgn=en_sgn/torch.sum(en_sgn)#normalize
        en_sgn=torch.reshape(en_sgn,(1,1,-1))
        en_bkg=self.bkg_en(input_en)+1
        
        if torch.sum(en_bkg)==0:
            en_bkg=1-en_bkg
        en_bkg=en_bkg/torch.sum(en_bkg)#normalize
        en_bkg=torch.reshape(en_bkg,(1,1,-1))

        total=sp_sgn*en_sgn+sp_bkg*en_bkg
            
        return sp_sgn,en_sgn,sp_bkg,en_bkg,total



    def poisson_loss(self):
        sp_sgn,en_sgn,sp_bkg,en_bkg,total=self.final_outputs()
        poisson_dist = dist.Poisson(total+m)
        loss = -poisson_dist.log_prob(self.obs)
        return torch.mean(loss)
