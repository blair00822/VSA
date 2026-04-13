# loss
import torch
import torch.nn as nn
import torch.nn.functional as F

def supervised_loss_out_pos_only(z: torch.Tensor,
                                 labels: torch.Tensor,
                                 tau: float = 0.07,
                                 eps: float = 1e-9) -> torch.Tensor:
    
    n   = z.size(0)
    dev = z.device
    labels = labels.view(-1, 1)              # (N,1)
   
    sim     = (z @ z.T) / tau                # (N,N)
    exp_sim = torch.exp(sim)                 # exp(sim/τ)

    diag         = torch.eye(n, device=dev)
    mask_anchor  = (labels == 1).float().squeeze(1)     # (N,) 
    mask_pos     = ((labels == 1) & (labels.T == 1)).float() - diag   
    mask_all     = 1.0 - diag                          

    denom   = (exp_sim * mask_all).sum(dim=1, keepdim=True)           # Σ_{a∈A(i)} exp
    log_prob = torch.log(exp_sim + eps) - torch.log(denom + eps)      

    loss_i = -(log_prob * mask_pos).sum(dim=1) / (mask_pos.sum(dim=1) + eps)

    loss = (loss_i * mask_anchor).sum() / (mask_anchor.sum() + eps)
    return loss

def noise_infonce_v4(feats, labels, model, sigma=0.1, tau=0.2):
    device = feats.device
    pos = labels == 1                          # (B,)
    if not pos.any():
        return feats.sum() * 0                      
    
    z_clean = feats[pos]                              # (N+,D)
    eps = torch.randn_like(z_clean)
    # z_noisy = √(1−σ²)·z_clean + σ·ε 
    z_noisy = (1.0 - sigma**2) ** 0.5 * z_clean + sigma * eps
    z_clean = model.fc(model.relu(z_clean))
    z_noisy = model.fc(model.relu(z_noisy))
    
    z_clean = F.normalize(z_clean, dim=1)
    z_noisy = F.normalize(z_noisy, dim=1)
    
    sim = (z_noisy @ z_clean.t()) / tau         # (N+,N+)
    target = torch.arange(sim.size(0), device=device)
    loss = F.cross_entropy(sim, target)
    return loss

def total_loss(mid_feats, feats, labels, model, sigma=0.05, lam1=0.1, lam2=0.1):
    # return ((1 - lam) * supervised_loss_out_pos_only(feats, labels)
    #         + lam * noise_infonce(mid_feats, labels, model, sigma))
    L1 = lam1 * supervised_loss_out_pos_only(feats, labels)
    L2 = lam2 * noise_infonce_v4(mid_feats, labels, model, sigma)
    L = L1 + L2
    return L, L1, L2
    
