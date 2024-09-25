# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:09:55 2024

@author: mario
"""


import pickle
from matplotlib import pyplot as plt
import numpy as np
import patchworklib as pw
from scipy.stats import poisson
import torch.distributions as dist
import torch
import io


def poisson_loss(total, target_data):
    poisson_dist = dist.Poisson(torch.tensor(total))
    loss = -poisson_dist.log_prob(torch.tensor(target_data))#*obs_weights
    return torch.mean(loss).data

def display_l(obs_data):
        
    obs=obs_data["observation"]
    name=obs_data["name"]

    if name==("simple_mock" or "mock_DM_signal"):
        total_sgn=obs_data["total_nopoisson"]
        target_loss_min=poisson_loss(total_sgn,obs)
        print("target L:",round(target_loss_min.item(),3))
    one_comp_e=np.mean(obs,axis=(0,1))
    one_comp_e=one_comp_e/np.sum(one_comp_e)
    one_comp_e=np.expand_dims(one_comp_e, axis=(0,1))

    one_comp_s=np.sum(obs,axis=2)
    one_comp_s=np.expand_dims(one_comp_s, axis=2)
    one_comp=one_comp_e*one_comp_s
    one_comp_loss=poisson_loss(one_comp,obs)
    overfit_loss_min=poisson_loss(obs,obs)
    print("one-component L:",round(one_comp_loss.item(),3))
    print("minimal (overfit) L:",round(overfit_loss_min.item(),3))


def display_stats(model,obs_data,loss_record,gu,n):
    
    obs=obs_data["observation"]
    e_scale=obs_data["e_scale"]
    s_scale=obs_data["s_scale"]
    name=obs_data["name"]

    if name==("simple_mock" or "mock_DM_signal"):
        total_sgn=obs_data["total_nopoisson"]
        target_loss_min=poisson_loss(total_sgn,obs)
    one_comp_e=np.mean(obs,axis=(0,1))
    one_comp_e=one_comp_e/np.sum(one_comp_e)
    one_comp_e=np.expand_dims(one_comp_e, axis=(0,1))

    one_comp_s=np.sum(obs,axis=2)
    one_comp_s=np.expand_dims(one_comp_s, axis=2)
    one_comp=one_comp_e*one_comp_s
    one_comp_loss=poisson_loss(one_comp,obs)


   
    ax_loss=pw.Brick(figsize=(6,4))
    ax_loss.loglog(gu,loss_record,label="loss",color="blue")
    ax_loss.loglog(gu,[one_comp_loss.cpu() for i in range(len(gu))],label="one-component L",color="red")
    
    if name==("simple_mock" or "mock_DM_signal"):
        ax_loss.loglog(loss_record,[target_loss_min.cpu() for i in range(len(loss_record))],label="ground truth L",color="black")
    ax_loss.set_title("Loss")
    ax_loss.legend(loc="upper right")
    ax_loss.set_xlabel("gradient updates")
    
    
    sp_sgn,en_sgn,sp_bkg,en_bkg,total=model.final_outputs()
    total=total.cpu().detach().numpy()

    ax_en=pw.Brick(figsize=(6,4))
    ax_en.set_title("estimation vs observation: energy")
    ax_en.semilogx(e_scale,np.sum(total, axis=(0,1)),label="estimation")
    ax_en.semilogx(e_scale,np.sum(obs, axis=(0,1)),label="observation")
    ax_en.legend(loc="upper right")
    ax_en.set_xlabel("Energy (TeV)")
    ax_en.set_ylabel("Total count")
    
    
    sp_est=np.sum(total,axis=2)
    sp_obs=np.sum(obs,axis=2)
    
    sp_est_1D=np.sum(total,axis=(1,2))
    sp_obs_1D=np.sum(obs,axis=(1,2))
    
    
    ax_sp_1D=pw.Brick(figsize=(6,4))
    ax_sp_1D.set_title("estimation vs observation: 1D space")
    ax_sp_1D.plot(s_scale[0,:,0], sp_est_1D,label="estimation")
    ax_sp_1D.plot(s_scale[0,:,0], sp_obs_1D,label="observation")
    ax_sp_1D.legend(loc="upper right")
    ax_sp_1D.set_xlabel("Position")
    ax_sp_1D.set_ylabel("Total count")
    
    ax_sp_est=pw.Brick(figsize=(4,4))
    ax_sp_est.set_title("estimated total signal")
    ax_sp_est.imshow(sp_est,cmap="rocket",vmin=min(np.min(sp_est),np.min(sp_obs)), vmax=max(np.max(sp_est),np.max(sp_obs)))
    ax_sp_est.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
   
    ax_sp_obs=pw.Brick(figsize=(4,4))
    ax_sp_obs.set_title("observation")
    ax_sp_obs.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax_sp_obs.imshow(sp_obs,cmap="rocket",vmin=min(np.min(sp_est),np.min(sp_obs)), vmax=max(np.max(sp_est),np.max(sp_obs)))

    
    ax_sp=ax_sp_obs|ax_sp_est
    ax_sp=ax_sp.add_colorbar(cmap="rocket",hratio=1,vmin=min(np.min(sp_est),np.min(sp_obs)), vmax=max(np.max(sp_est),np.max(sp_obs)))
    
#    ax_all=(ax_loss|ax_en)/ax_sp
    ax_all=ax_loss|ax_en|ax_sp_1D
    ax_all.set_suptitle("Stats, try n°"+str(n),fontsize=30)

 
    buf = io.BytesIO()
    ax_all.savefig(buf, format='png')
    buf.seek(0)

    # Display using matplotlib
    img = plt.imread(buf)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def compute_stats(model,obs,loss_record):
    

    sp_src,en_src,sp_bkg,en_bkg,total=model.final_outputs()
    sp_src=sp_src.cpu().detach().numpy()
    en_src=en_src.cpu().detach().numpy()
    sp_bkg=sp_bkg.cpu().detach().numpy()
    en_bkg=en_bkg.cpu().detach().numpy()
    total=total.cpu().detach().numpy()
    
    
    obs_weights=np.where(obs==0,1,obs)

    obs_weights=np.where(obs==0,1,obs)
    weighted_logprob=np.sum(obs_weights*poisson.logpmf(obs,total))/np.sum(obs_weights)

    weighted_prob=np.exp(weighted_logprob)

        

    return(np.array([np.squeeze(sp_src)]),np.array([np.squeeze(en_src)]),np.array([np.squeeze(sp_bkg)]),np.array([np.squeeze(en_bkg)]),np.array([loss_record]),np.array([weighted_prob]))

def build_and_save_ensemble_stats(specs,obs_data,all_stats,gu):
    
    nruns,nepochs,lr,sp_hat,med_block,name=specs
    e_scale,s_scale=obs_data["e_scale"],obs_data["s_scale"]
    all_s_src,all_e_src,all_s_bkg,all_e_bkg,all_loss,w_probs=all_stats
    savename=name
    if sp_hat=="soft":
        savename+="_softhat"
    elif sp_hat==True:
        savename+="_hat"
    if med_block==True:
        savename+="_medblock"
    weights=w_probs/np.sum(w_probs)


    #compute weighted averages for e and s components
    
    e_weights_reshape=np.expand_dims(weights, 1)
    
    avg_e_src=np.sum(all_e_src*e_weights_reshape,axis=0)
    std_e_src_0=np.sqrt((all_e_src-np.expand_dims(avg_e_src,0))**2)
    std_e_src=np.sum(std_e_src_0*e_weights_reshape,axis=0)
    
    avg_e_bkg=np.sum(all_e_bkg*e_weights_reshape,axis=0)
    std_e_bkg_0=np.sqrt((all_e_bkg-np.expand_dims(avg_e_bkg,0))**2)
    std_e_bkg=np.sum(std_e_bkg_0*e_weights_reshape,axis=0)


    s_weights_reshape=np.expand_dims(weights, (1,2))
    
    avg_s_src=np.sum(all_s_src*s_weights_reshape,axis=0)
    std_s_src_0=np.sqrt((all_s_src-np.expand_dims(avg_s_src,0))**2)
    std_s_src=np.sum(std_s_src_0*s_weights_reshape,axis=0)
    
    avg_s_bkg=np.sum(all_s_bkg*s_weights_reshape,axis=0)
    std_s_bkg_0=np.sqrt((all_s_bkg-np.expand_dims(avg_s_bkg,0))**2)
    std_s_bkg=np.sum(std_s_bkg_0*s_weights_reshape,axis=0)



    #Store everything in a dictionary
    data = {
        "all_E_src": all_e_src,
        "all_S_src": all_s_src,
        "all_E_bkg": all_e_bkg,
        "all_S_bkg": all_s_bkg,
        "avg_E_src": avg_e_src,
        "avg_S_src": avg_s_src,
        "avg_E_bkg": avg_e_bkg,
        "avg_S_bkg": avg_s_bkg,
        "std_E_src": std_e_src,
        "std_S_src": std_s_src,
        "std_E_bkg": std_e_bkg,
        "std_S_bkg": std_s_bkg,
        "e_scale": e_scale,
        "s_scale": s_scale,    
        "nruns": nruns,
        "gradient updates": gu,
        "all_loss": all_loss,
        "name": name,
        "savename":savename,
        "lr": lr,
        "nepochs": nepochs,
        "med_block": med_block,
        "sp_hat": sp_hat,
        "w_probs":w_probs,
        "weights":weights

    }


    pickle.dump(data,open("save_data/stats_"+savename+'.pkl', 'wb'))
    return(data)

def display_ensemble_stats(save_data,obs_data):#plot ensemble stats after n runs
########################
    e_scale=obs_data["e_scale"]
    true_e_src=obs_data["E_src"]
    true_e_bkg=obs_data["E_bkg"]
    s_scale=obs_data["s_scale"]
    s_scale_1D=s_scale[0,:,0]
    true_s_src=obs_data["S_src"]
    true_s_bkg=obs_data["S_bkg"]
    true_s_src_x=np.sum(true_s_src,axis=0)
    true_s_bkg_x=np.sum(true_s_bkg,axis=0)

    
    
    all_e_src=save_data["all_E_src"]
    all_e_bkg=save_data["all_E_bkg"]
    
    all_s_src=save_data["all_S_src"]
    all_s_bkg=save_data["all_S_bkg"]
    all_s_src_x=np.sum(all_s_src,axis=1)
    all_s_bkg_x=np.sum(all_s_bkg,axis=1)

    avg_e_src=save_data["avg_E_src"]
    avg_e_bkg=save_data["avg_E_bkg"]    
    avg_s_src=save_data["avg_S_src"]
    avg_s_bkg=save_data["avg_S_bkg"]
    avg_s_src_x=np.sum(avg_s_src,axis=0)
    avg_s_bkg_x=np.sum(avg_s_bkg,axis=0)

    std_e_src=save_data["std_E_src"]
    std_e_bkg=save_data["std_E_bkg"]    
    std_s_src=save_data["std_S_src"]
    std_s_bkg=save_data["std_S_bkg"]
    std_s_src_x=np.sum(std_s_src,axis=0)
    std_s_bkg_x=np.sum(std_s_bkg,axis=0)
    
    gu=save_data["gradient updates"]
    all_loss=save_data["all_loss"]

    n=save_data["nruns"]
    savename=save_data["savename"]
    
    reds = plt.cm.plasma(np.linspace(0.2, 1.0, n))
    blues=plt.cm.viridis(np.linspace(0.2, 1.0, n))
    
########################plot individual estimations    
    #display energy component estimations for all runs
    axen_all=pw.Brick(figsize=(6,4))
    for i in range(n):
        axen_all.semilogx(e_scale,all_e_src[i],color=reds[i],linestyle="dotted")
        axen_all.semilogx(e_scale,all_e_bkg[i],color=blues[i],linestyle="dotted")
    axen_all.semilogx(e_scale,true_e_src,label="Source: GT",color="red")
    axen_all.semilogx(e_scale,true_e_bkg,label="Background: GT",color="blue")
    axen_all.plot([], [], color=reds[4], linestyle="dotted", label="Source: NN")
    axen_all.plot([], [], color=blues[4], linestyle="dotted", label="Background: NN")
    axen_all.set_ylabel("Counts (normalized)")
    axen_all.set_xlabel("Energy (TeV)")
    axen_all.set_title("Energy Spectra")
    axen_all.legend(loc="upper right")


    
    #display 1D space component estimations for all runs
    ax_sp_x_all=pw.Brick(figsize=(6,4))
    ax_sp_x_all.plot(s_scale_1D,true_s_src_x,label="Source: GT",color="red")
    ax_sp_x_all.plot(s_scale_1D,true_s_bkg_x,label="Background: GT",color="blue")
    for i in range(n):
        ax_sp_x_all.plot(s_scale_1D,all_s_src_x[i],color=reds[i],linestyle="dotted")#,label=str(i))
        ax_sp_x_all.plot(s_scale_1D,all_s_bkg_x[i],color=blues[i],linestyle="dotted")#,label=str(i))
    ax_sp_x_all.plot([], [], color=reds[4], linestyle="dotted", label="Source: NN")
    ax_sp_x_all.plot([], [], color=blues[4], linestyle="dotted", label="Background: NN")
    ax_sp_x_all.set_ylabel("Counts")
    ax_sp_x_all.set_xlabel("Position")
    ax_sp_x_all.set_title("Space distribution - single axis")
    ax_sp_x_all.legend(loc="upper right")

    
    # all source space estimations - 2D colormaps
    n2=int(n/2)
    pw.param["margin"]=0.2
    axim_src1=pw.Brick(figsize=(1,1))
    axim_src1.imshow(all_s_src[0],vmin=np.min(all_s_src), vmax=np.max(all_s_src),cmap="rocket")
    axim_src1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    axim_src2=pw.Brick(figsize=(1,1))
    axim_src2.imshow(all_s_src[n2],vmin=np.min(all_s_src), vmax=np.max(all_s_src),cmap="rocket")
    axim_src2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    for i in range(1,n2):
        axn1=pw.Brick(figsize=(1,1))
        axn1.imshow(all_s_src[i],vmin=np.min(all_s_src), vmax=np.max(all_s_src),cmap="rocket")
        axn1.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        axim_src1=axim_src1|axn1
        
        axn2=pw.Brick(figsize=(1,1))
        axn2.imshow(all_s_src[i+n2],vmin=np.min(all_s_src), vmax=np.max(all_s_src),cmap="rocket")
        axn2.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
        axim_src2=axim_src2|axn2
    axim_srcbis=axim_src1/axim_src2    
    axim_srcbis=axim_srcbis.add_colorbar(cmap="rocket",hratio=1,vmin=np.min(all_s_src), vmax=np.max(all_s_src))
    axim_srcbis.set_suptitle('Estimated Source',fontsize=22,)

###########################  average estimations  
    # average space source estimation
    src_m,src_M=np.min(avg_s_src),np.max(avg_s_src)
    ax_sp_src_nn=pw.Brick(figsize=(4,4))
    ax_sp_src_nn.imshow(avg_s_src,cmap="rocket",vmin=src_m, vmax=src_M)
    ax_sp_src_nn.set_title("Average Source: NN")#" ("+name+")")#,fontsize=15)
    ax_sp_src_nn.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax_sp_src_nn=ax_sp_src_nn.add_colorbar(cmap="rocket",hratio=1,vmin=src_m, vmax=src_M)

    #average energy components estimation
    axen_avg=pw.Brick(figsize=(6,4))
    axen_avg.semilogx(e_scale,true_e_src,label="Source: GT",color="red")
    axen_avg.semilogx(e_scale,true_e_bkg,label="Background: GT",color="blue")
    axen_avg.semilogx(e_scale,avg_e_src,label="Source: NN",color="pink")
    axen_avg.semilogx(e_scale,avg_e_bkg,label="Background: NN",color="lightblue")
    axen_avg.fill_between(e_scale, avg_e_src-std_e_src,avg_e_src+std_e_src,color="pink",alpha=.3,label="Source std: NN")
    axen_avg.fill_between(e_scale, avg_e_bkg-std_e_bkg,avg_e_bkg+std_e_bkg,color="lightblue",alpha=.3,label="Background std: NN")
    axen_avg.set_ylabel("Counts (normalized)")
    axen_avg.set_xlabel("Energy (TeV)")
    axen_avg.set_title("Energy Spectra")
    axen_avg.legend(loc="upper right")#,fontsize=16)

    # average space source estimation
    ax_sp_x_avg=pw.Brick(figsize=(6,4))
    ax_sp_x_avg.plot(s_scale_1D,true_s_src_x,label="Source: GT",color="red")
    ax_sp_x_avg.plot(s_scale_1D,true_s_bkg_x,label="Background: GT",color="blue")
    ax_sp_x_avg.plot(s_scale_1D,avg_s_src_x,label="Source: NN",color="firebrick")
    ax_sp_x_avg.plot(s_scale_1D,avg_s_bkg_x,label="Background: NN",color="steelblue")
    #ax_sp_x_avg.fill_between(s_scale_1D, w_avg_src_x-excess_std_x,w_avg_src_x+excess_std_x,color="orange",alpha=.3,label="Poisson std: NN")
    ax_sp_x_avg.fill_between(s_scale_1D, avg_s_src_x-std_s_src_x,avg_s_src_x+std_s_src_x,color="firebrick",alpha=.3,label="Source std: NN")
    ax_sp_x_avg.fill_between(s_scale_1D, avg_s_bkg_x-std_s_bkg_x,avg_s_bkg_x+std_s_bkg_x,color="steelblue",alpha=.3,label="Background std: NN")
    ax_sp_x_avg.set_ylabel("Counts")
    ax_sp_x_avg.set_xlabel("position")
    ax_sp_x_avg.set_title("Space distribution - single axis")
    ax_sp_x_avg.legend(loc="upper right")


###########################    
    #plot losses
    ax_loss = pw.Brick(figsize=(6,4))
    ax_loss.set_title("Losses")
    for i in range(n):
        ax_loss.plot(gu,all_loss[i],color=reds[i])
    ax_loss.set_xlabel("Gradient updates")
    ax_loss.set(xscale='log',yscale='log')
    
    savedir="save_data/"
    # summary for independent runs
    ind_run_ax=(axen_all|ax_sp_x_all)/(ax_loss|axim_srcbis)
    #ind_run_ax.set_suptitle("Individual Runs: "+name,fontsize=30)
    ind_run_ax.savefig(savedir+savename+"_all.jpeg")
    
    #summary for averaged runs
    avg_run_ax=(ax_sp_src_nn|axen_avg|ax_sp_x_avg)
#    avg_run_ax.set_suptitle("Averaged Runs: "+name,fontsize=30)
    avg_run_ax.savefig(savedir+savename+"_avg.jpeg")

    buf = io.BytesIO()
    ind_run_ax.savefig(buf, format='png')
    buf.seek(0)

    # Display using matplotlib
    img = plt.imread(buf)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()
    
    buf = io.BytesIO()
    avg_run_ax.savefig(buf, format='png')
    buf.seek(0)

    # Display using matplotlib
    img = plt.imread(buf)
    plt.imshow(img)
    plt.axis('off')
    plt.show()