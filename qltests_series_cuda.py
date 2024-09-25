# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:18:35 2023

@author: mario
"""

import pickle
from matplotlib import pyplot as plt
import numpy as np
import patchworklib as pw
import seaborn as sns
import pandas as pd
from scipy.stats import poisson
import torch.distributions as dist
import torch

def poisson_loss(total, target_data):
    poisson_dist = dist.Poisson(torch.tensor(total))
    loss = -poisson_dist.log_prob(torch.tensor(target_data))#*obs_weights
    return torch.mean(loss).data

def display_stats(name,model,data,loss_record,n):
    
    obs=data["observation"]
    e_scale=data["e_scale"]
    #name=data["name"]

    if name==("simple_mock" or "mock_DM_signal"):
        total_sgn=data["total_nopoisson"]
        target_loss_min=poisson_loss(total_sgn,obs)
        print("target loss:",target_loss_min)
    one_comp_e=np.mean(obs,axis=(0,1))
    one_comp_e=one_comp_e/np.sum(one_comp_e)
    one_comp_e=np.expand_dims(one_comp_e, axis=(0,1))

    one_comp_s=np.sum(obs,axis=2)
    one_comp_s=np.expand_dims(one_comp_s, axis=2)
    one_comp=one_comp_e*one_comp_s
    one_comp_loss=poisson_loss(one_comp,obs)
    overfit_loss_min=poisson_loss(obs,obs)
    print("one-component loss:",one_comp_loss)
    print("minimal loss:",overfit_loss_min)


    
#axs[0, 0].plot(x, y)
#axs[0, 0].set_title('Axis [0, 0]')
#axs[0, 1].plot(x, y, 'tab:orange')
#axs[0, 1].set_title('Axis [0, 1]')
#axs[1, 0].plot(x, -y, 'tab:green')
#axs[1, 0].set_title('Axis [1, 0]')
#axs[1, 1].plot(x, -y, 'tab:red')
#axs[1, 1].set_title('Axis [1, 1]')

#for ax in axs.flat:
 #   ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
 #   ax.label_outer()

#%% 
"""
    fig, axs = plt.subplots(2, 2)   
    axs[0, 0].loglog(loss_record[0],loss_record[1])
    axs[0, 0].plot(loss_record[0],[one_comp_loss.cpu() for i in range(len(loss_record[0]))],label="one-component loss",color="red")
    
    if name==("simple_mock" or "mock_DM_signal"):
        axs[0, 0].semilogy(loss_record[0],[target_loss_min.cpu() for i in range(len(loss_record[0]))],label="perfect loss",color="black")
    axs[0, 0].set_title("Loss, try n°"+str(n))
    axs[0, 0].legend(loc="upper right")
    axs[0, 0].set_xlabel("gradient updates")

    sp_sgn,en_sgn,sp_bkg,en_bkg,total=model.final_outputs()
    total=total.cpu().detach().numpy()
    
    #plt.plot()
    axs[0, 1].set_title("estimated total energy vs observed total energy")
    axs[0, 1].semilogx(e_scale,np.sum(total, axis=(0,1)),label="estimation")
    axs[0, 1].semilogx(e_scale,np.sum(obs, axis=(0,1)),label="observation")
    axs[0, 1].legend(loc="upper right")
    axs[0, 1].set_xlabel("Energy (TeV)")
    axs[0, 1].set_ylabel("Total count")
    #plt.show()
    #plt.close()    
    

    #plt.plot()
    axs[1,0].set_title("estimated total signal, try n°"+str(n))
    axs[1,0].imshow(np.sum(total,axis=2))
    axs[1,0].colorbar()
    #plt.show()
    #plt.close()
    
    #plt.plot()
    axs[1,1].set_title("observation, try n°"+str(n))
    axs[1,1].imshow(np.sum(obs,axis=2))
    axs[1,1].colorbar()
    
    plt.show()
    plt.close()
    
"""
#%%   

    plt.loglog(loss_record[0],loss_record[1])
    plt.plot(loss_record[0],[one_comp_loss.cpu() for i in range(len(loss_record[0]))],label="one-component loss",color="red")
    
    if name==("simple_mock" or "mock_DM_signal"):
        plt.semilogy(loss_record[0],[target_loss_min.cpu() for i in range(len(loss_record[0]))],label="perfect loss",color="black")
    plt.title("Loss, try n°"+str(n))
    plt.legend(loc="upper right")
    plt.xlabel("gradient updates")
    plt.show()
    plt.close()
    
    
    sp_sgn,en_sgn,sp_bkg,en_bkg,total=model.final_outputs()
    #sp_sgn=sp_sgn.cpu().detach().numpy()
    #en_sgn=en_sgn.cpu().detach().numpy().reshape(-1)
    #sp_bkg=sp_bkg.cpu().detach().numpy()
    #en_bkg=en_bkg.cpu().detach().numpy().reshape(-1)
    total=total.cpu().detach().numpy()

    plt.plot()
    plt.title("estimated total signal, try n°"+str(n))
    plt.imshow(np.sum(total,axis=2))
    plt.colorbar()
    plt.show()
    plt.close()
    
    plt.plot()
    plt.title("observation, try n°"+str(n))
    plt.imshow(np.sum(obs,axis=2))
    plt.colorbar()
    plt.show()
    plt.close()
    
    plt.plot()
    plt.title("estimated total energy vs observed total energy")
    plt.semilogx(e_scale,np.sum(total, axis=(0,1)),label="estimation")
    plt.semilogx(e_scale,np.sum(obs, axis=(0,1)),label="observation")
    plt.legend(loc="upper right")
    plt.xlabel("Energy (TeV)")
    plt.ylabel("Total count")
    plt.show()
    plt.close()


#%%
def compute_stats(model,obs,loss_record):

    sp_src,en_src,sp_bkg,en_bkg,total=model.final_outputs()
    sp_src=sp_src.cpu().detach().numpy()
    en_src=en_src.cpu().detach().numpy()
    sp_bkg=sp_bkg.cpu().detach().numpy()
    en_bkg=en_bkg.cpu().detach().numpy()
    total=total.cpu().detach().numpy()
    
    
    tot=sp_src*en_src+sp_bkg*en_bkg


    obs_weights=np.where(obs==0,1,obs)

    weighted_logprob=np.sum(obs_weights*np.log(poisson.pmf(obs,tot)))
    weighted_meanlogprob=np.exp(weighted_logprob)
    

    return(np.array([sp_src]),np.array([en_src]),np.array([sp_bkg]),np.array([en_bkg]),np.array([loss_record]),np.array([weighted_meanlogprob]))



