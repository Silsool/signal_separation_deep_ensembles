import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from torch import optim
import numpy as np
import pickle

from NN_archis import NN_archi
import qltests as qlt
import time

start_time = time.time()

#%%

lr=0.01 # 0.001 by default, .01 for mock_DM_signal


nruns=10 # number of runs to average
nepochs =100000 #base is 100000
showtimes=2 # to look at stats during run, optional
sp_hat="soft" #True, False or "soft"
med_block=True #True or False

name="mock_DM_signal"#"MSH15_52","Crab","simple_mock" or "mock_DM_signal"
name_plus=""

obs_data=pickle.load(open('obs_data/'+name+'.pkl', 'rb'))
obs=obs_data["observation"]

specs=nruns,nepochs,lr,sp_hat,med_block,name,name_plus


qlt.display_l(obs_data)#display (-)log-likelihoods of different scenarios - to compare with loss
#%%
# Training loop

    
for n in range(nruns):
    
    
    #initialize model weights
    model = NN_archi(obs,sp_hat=sp_hat,med_block=med_block)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_record=[]
    gu=[]
    
    for epoch in range(nepochs):
               
        # Compute the loss
        loss = model.poisson_loss()
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # Save loss 100 times per run
        if (epoch + 1) % int(nepochs/100) == 0:
            loss_record=np.append(loss_record,loss.data.cpu())
            gu=np.append(gu,epoch+1)
            
        # Plot stats x times per run, optional
        if (epoch + 1) % int(nepochs/showtimes) == 0:
            print(f"Run [{n+1}/{nruns}], Epoch [{epoch+1}/{nepochs}], Loss: {round(loss.item(),3)}")
            qlt.display_stats(model, obs_data, loss_record,gu,n)
        
        
            
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")       
    # Compute stats after every run
    current_stats=qlt.compute_stats(model,obs,loss_record)
    if n==0:
        all_stats=current_stats
    else:
        (all_src_sp,all_src_en,all_bkg_sp,all_bkg_en,all_loss,all_prob)=(np.append(all_stats[i],current_stats[i],axis=0) for i in range(len(all_stats)))
        all_stats=(all_src_sp,all_src_en,all_bkg_sp,all_bkg_en,all_loss,all_prob)

#%%
#Compute, display and save ensemble stats

save_data=qlt.build_and_save_ensemble_stats(specs,obs_data,all_stats,gu)
#save_data=pickle.load(open('save_data/stats_mock_DM_signal.pkl', 'rb'))

qlt.display_ensemble_stats(save_data,obs_data)
