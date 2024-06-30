import numpy as np
from scipy.signal import savgol_filter
from scipy import linalg
import torch 


# parallelized with matrix operations
# 0.2 seconds
def SVD_approximation_inverse_Laplace(config, time_horizon, Q_gamma, device='cpu'):
    """
    SVD-based approximation of the inverse Laplace transform 
    Input:
    config: configuration object from config.py
    time_horizon: new time horizon
    Q_gamma: Q-values for different gamma values and sensitivities, 
             shape (batch_size, num_gamma, num_sensitivities)
    """
    alpha_reg = config.alpha_reg
    # K = config.K
    delta_t = config.delta_t
    
    #batch_size = config.BATCH_SIZE # NOTE maybe not at the beginning of the training - then assume change in time horizon takes place after that
    batch_size = Q_gamma.shape[0]
    num_sensitivities = config.num_sensitivities
    num_actions = config.action_dim
    num_gamma_to_tau = config.num_gamma_to_tau
    gamma_to_tau_min = config.gamma_to_tau_min
    gamma_to_tau_max = config.gamma_to_tau_max
    
    start = 1 / np.log(gamma_to_tau_min) 
    end = 1 / np.log(gamma_to_tau_max)   
    gammas_to_tau = torch.exp(torch.true_divide(1, torch.linspace(start, end, num_gamma_to_tau)))
    
    assert Q_gamma.shape == (batch_size, num_actions, num_gamma_to_tau, num_sensitivities), "Q_gamma shape does not match (num_gamma, num_sensitivities)"
    # finish extending to actions

    #define matrix F:
    F = torch.zeros((len(gammas_to_tau), time_horizon))
    gammas_to_tau_tensor = torch.tensor(gammas_to_tau)
    delta_t_tensor = torch.tensor(delta_t)

    # Perform matrix operations instead of loops
    F = gammas_to_tau_tensor.unsqueeze(1).pow(torch.arange(time_horizon, device=device).float() * delta_t_tensor).to(device)

    U, lam, V = torch.linalg.svd(F) #SVD decomposition of F

    # set up gamma-space:
    Z = Q_gamma[:,:,:,0:-2]-Q_gamma[:,:,:,1:-1]
    Z = Z.to(device)

    # smooth gamma-space (it might not be necessary, it helps if the input is *very* noisy):
    # for h in range(0,num_h-2):
    #     Z[:,h]=savgol_filter(Z[:,h], 5, 1)

    # Linearly recover tau-space from eigenspace of F:
    fi = lam**2 / (alpha_reg**2 + lam**2)
    # Expand tensors for broadcasting
    fi = fi.reshape(1, 1, -1, 1)  # Shape: (1, 1, len(lam), 1)
    fi = fi.to(device)

    U_expanded = U.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, K, len(lam))
    V_expanded = V.unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, len(lam), K)
    lam = lam.to(device)
    
    tau_space=torch.zeros((batch_size, num_actions, time_horizon, num_sensitivities-2)).to(device)
    for h in range(num_sensitivities - 2):
        Z_expanded = Z[:, :, :, h].unsqueeze(2)

        tmp = (Z_expanded @ U_expanded).permute(0,1,3,2)
        V_lam = V_expanded[:,:,:len(lam),:] 
        term = (fi * (tmp * V_lam) / lam.reshape(1, 1, -1, 1)).sum(dim=2)
        tau_space[:, :, :, h] = term
        
            
    #smooth tau-space (it might not be necessary, use for a smoother visualization):
    # for h in range(0,num_h-2):
    #     tau_space[:,h]=savgol_filter(tau_space[:,h], 11, 1)

    #Normalization (it is not really necessary for this very short temporal horizon T=4):
    tau_space[tau_space < 0] = 0

    sum_tau = torch.nansum(tau_space, dim=-1, keepdim=True)
    mask = sum_tau > 0
    sum_tau[sum_tau == 0] = 1

    tau_space = tau_space / sum_tau

    mask_expanded = mask.expand_as(tau_space)
    tau_space[~mask_expanded] = 0
            
    return tau_space


# Not parallelized, iterative approach
# 8.5 seconds

def SVD_approximation_inverse_Laplace_iterative(config, Q_gamma):
    """
    SVD-based approximation of the inverse Laplace transform 
    Input:
    config: configuration object from config.py
    Q_gamma: Q-values for different gamma values and sensitivities, 
             shape (batch_size, num_gamma, num_sensitivities)
    """
    alpha_reg = config.alpha_reg
    K = config.K
    delta_t = config.delta_t
    
    #batch_size = config.BATCH_SIZE # NOTE maybe not at the beginning of the training - then assume change in time horizon takes place after that
    batch_size = Q_gamma.shape[0]
    num_sensitivities = config.num_sensitivities
    num_actions = config.action_dim
    num_gamma_to_tau = config.num_gamma_to_tau
    gamma_to_tau_min = config.gamma_to_tau_min
    gamma_to_tau_max = config.gamma_to_tau_max
    start = 1 / np.log(gamma_to_tau_min) 
    end = 1 / np.log(gamma_to_tau_max)   
    gammas_to_tau = torch.exp(torch.true_divide(1, torch.linspace(start, end, num_gamma_to_tau)))
    
    assert Q_gamma.shape == (batch_size, num_actions, num_gamma_to_tau, num_sensitivities), "Q_gamma shape does not match (num_gamma, num_sensitivities)"
    # finish extending to actions

    #define matrix F:
    F=torch.zeros((len(gammas_to_tau),K))
    for i_g in range(0,len(gammas_to_tau)):
        for i_t in range(0,K):
            F[i_g,i_t]=gammas_to_tau[i_g]**(i_t*delta_t)

    U, lam, V = linalg.svd(F) #SVD decomposition of F

    # set up gamma-space:
    Z=Q_gamma[:,:,:,0:-2]-Q_gamma[:,:,:,1:-1]

    # smooth gamma-space (it might not be necessary, it helps if the input is *very* noisy):
    # for h in range(0,num_h-2):
    #     Z[:,h]=savgol_filter(Z[:,h], 5, 1)

    # Linearly recover tau-space from eigenspace of F:
    tau_space=torch.zeros((batch_size, num_actions, K, num_sensitivities-2))
    # do for several batches and actions-> parallelize TODO
    for batch in range(batch_size):
        for act in range(num_actions):
            for h in range(0,num_sensitivities-2):
                term=torch.zeros((1,K))
                for i in range(0,len(lam)):
                    fi=lam[i]**2/(alpha_reg**2+lam[i]**2)
                    new=fi*(((U[:,i]@Z[batch,act,:,h])*V[i,:] )/lam[i])
                    term=term+new
                tau_space[batch,act,:,h]=term
            
    #smooth tau-space (it might not be necessary, use for a smoother visualization):
    # for h in range(0,num_h-2):
    #     tau_space[:,h]=savgol_filter(tau_space[:,h], 11, 1)

    #Normalization (it is not really necessary for this very short temporal horizon T=4):
    tau_space[tau_space<0]=0 #make all probabilities positive
    # do for several batches and actions -> parallelize TODO
    for batch in range(batch_size):
        for a in range(num_actions):
            for i in range(0,K):
                if torch.nansum(tau_space[batch,a,i,:])>0.0:
                    tau_space[batch,a,i,:]=tau_space[batch,a,i,:]/torch.nansum(tau_space[batch,a,i,:])
            
    return tau_space