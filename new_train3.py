'''
only visual modality

for AVLips

'''
from new_supcon import total_loss
import torch
from torch import optim
from torch.utils.data import DataLoader
import os, csv, random, logging, time
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from new_dataset_v import AVLips_augmixmix 
from fairseq import checkpoint_utils
from fairseq import utils as fairseq_utils  
from utils import set_seed, get_logger

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu', help='gpu id to use', default='0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, help='epoch num when training', default=200)
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--data_path',type=str, 
                        default='./AVLips_test_file96/AVLips_real.csv', help='Path to data') ##
    parser.add_argument('--ckpt_path', default='./model_hubert/self_large_vox_433h.pt') 
    parser.add_argument('--save_prefix', type=str, 
                        default='./model_hubert/avlips_main_v4', help='Path to save checkpoint') ##
    parser.add_argument('--obj', type=str, default='ID_0', help='choose specific user model')
    parser.add_argument('--backbone_out_dim', type=int, default=256)
    parser.add_argument('--projector_feat_dim', type=int, default=128)
    parser.add_argument('--train_num', type=int, default=500)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--lam1', type=float, default=1)
    parser.add_argument('--lam2', type=float, default=1)
    return parser.parse_args()

class AVHuVisualPreprocess(torch.nn.Module):
    def __init__(self, image_crop_size=88, image_mean=0.421, image_std=0.165):
        super().__init__()
        self.size = image_crop_size
        self.register_buffer("mean", torch.tensor(image_mean, dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(image_std,  dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x):  
        B, C, T, H, W = x.shape
        x = x * 255.0

        top = (H - self.size) // 2
        left = (W - self.size) // 2
        x = x[:, :, :, top:top+self.size, left:left+self.size]  # (B,1,T,S,S)

        x = (x - self.mean) / self.std
        return x

class FrozenAVHuBackbone(torch.nn.Module):
    def __init__(self, avhu_model, feat_dim_out=None, device="cuda:0"):
    
        super().__init__()
        self.avhu = avhu_model.eval()
        for p in self.avhu.parameters():
            p.requires_grad = False  # 显式冻结

        self.device = device
        self.feat_dim_out = feat_dim_out
        self.proj = None

        D = 1024
        if feat_dim_out is not None and feat_dim_out != D:
            self.proj = torch.nn.Linear(D, feat_dim_out)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        self.out_dim = feat_dim_out or D

    def forward(self, x):
    
        with torch.no_grad():
            feat, _ = self.avhu.extract_finetune(
                    source={"video": x, "audio": None},
                    padding_mask=None, output_layer=None
                )  # (B, T, D)
           
        feat = feat.mean(dim=1)  # (B, D) 
        if self.proj is not None:
            feat = self.proj(feat)  # (B, D'), D' = feat_dim_out
        return feat
    
class Projector(nn.Module): # mid DM
    def __init__(self, backbone, in_dim=256, feat_dim=128):
        super(Projector, self).__init__()
        self.backbone = backbone
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(in_dim, feat_dim)
        self.batch_norm = nn.BatchNorm1d(in_dim, affine=False)
        
    def forward(self, x, return_features=False):
        out = self.backbone(x)           # FrozenAVHuBackbone.forward()
        out1 = self.batch_norm(out)      # Add BatchNorm for GDM
        if return_features:
            return out                   # (B, in_dim)
        out2 = self.relu(out)            
        out2 = self.fc(out2)             # (B, feat_dim)
        out2 = F.normalize(out2, dim=-1) #
        return out1, out2
    
def _load_projector_only(model, state):
    saved = state['state_dict']
    cur_sd = model.state_dict()
    remap = {}
    has_module = any(k.startswith("module.") for k in cur_sd.keys())
    prefix = "module." if has_module else ""

    for k, v in saved.items():
        target_k = prefix + k
        if target_k in cur_sd and cur_sd[target_k].shape == v.shape:
            remap[target_k] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    return missing, unexpected
   
    
def get_model(avhubert_model, device, 
              image_crop_size, image_mean, image_std,
              backbone_out_dim=256, projector_feat_dim=128,
              checkpoint_path=None):
    preprocess = AVHuVisualPreprocess(
            image_crop_size=image_crop_size,
            image_mean=image_mean,
            image_std=image_std
        ).to(device)
    frozen_backbone = FrozenAVHuBackbone(
            avhu_model=avhubert_model.to(device),
            feat_dim_out=backbone_out_dim,
            device=device
        )
    model = Projector(
            backbone=frozen_backbone,
            in_dim=backbone_out_dim,
            feat_dim=projector_feat_dim,
        ).to(device)    
    
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location = 'cpu') # .pkl
        missing, unexpected = _load_projector_only(model, state) 
        print("missing:", missing)
        print("unexpected:", unexpected)
    
    return preprocess, model

def _extract_projector_state(full_state_dict, include_avhu_bn=True):
    keep = {}
    for k, v in full_state_dict.items():
        k0 = k[7:] if k.startswith("module.") else k
        if (k0.startswith("batch_norm.") or
            k0.startswith("fc.") or
            k0.startswith("backbone.proj.")):
            keep[k0] = v
        if include_avhu_bn and k0.startswith("backbone.avhu"):
            if (k0.endswith(".running_mean") or
                k0.endswith(".running_var") or
                k0.endswith(".num_batches_tracked")):
                keep[k0] = v
    return keep

def train(args, avhu_model, task, dataset, device, save_path):
    
    preprocess, model = get_model(
        avhubert_model=avhu_model,
        device=device,
        image_crop_size=task.cfg.image_crop_size,
        image_mean=task.cfg.image_mean,
        image_std=task.cfg.image_std,
        backbone_out_dim=args.backbone_out_dim,
        projector_feat_dim=args.projector_feat_dim,
        checkpoint_path=None
    )
    model.train()
  
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_num = len(dataset)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    log_info = time.strftime(os.path.join('log_info_v4', args.obj + '_log_info_%m%d_%H:%M:%S.log'))
    os.makedirs('log_info_v4', exist_ok=True)
    logger = get_logger(log_info)
    logger.info("====== All args ======")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    logger.info("====== Start Training ======")
    
    train_loss_list = []
    count = 0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = []
        L_supcon, L_gdm = [], []
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for video_list, label_list in progress:
            B = video_list[0].size(0)
            x = torch.cat(video_list, dim=0).to(device) 
            labels = torch.cat(label_list, dim=0).to(device)
        
            with torch.no_grad():        
                x = preprocess(x)        # AVHuVisualPreprocess.forward()
            
            mid_features, features = model(x)  # (3B, feat_dim) or (6B, feat_dim)
            
            loss, L1, L2 = total_loss(mid_features, features, labels, model, sigma=args.sigma, lam1=args.lam1, lam2=args.lam2)
                      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item()) 
            L_supcon.append(L1.item())
            L_gdm.append(L2.item())
            progress.set_postfix(loss=loss.item())

        avg_loss = np.mean(train_loss)
        avg_l1 = np.mean(L_supcon)
        avg_l2 = np.mean(L_gdm)
        end = time.time()
        
        logger.info(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Loss1: {avg_l1:.6f} Loss2: {avg_l2:.8f}| Time: {(end-start):.6f}")
       
        if len(train_loss_list) == 0 or avg_loss < min(train_loss_list):
            count = 0 
            logger.info("saving best model ...")
            full_sd = model.state_dict()
            light_sd = _extract_projector_state(full_sd, include_avhu_bn=True)
            torch.save({'state_dict': light_sd}, os.path.join(save_path, 'checkpoint.pkl'))
             
        else:
            count += 1
        train_loss_list.append(avg_loss)
        
        if count == 10:
            logger.info("End of Training")
            break
   
if __name__ == "__main__":
    args = arg_parse()
    set_seed(42)
    print('training user:', args.obj)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
   
    dataset = AVLips_augmixmix('train', data_path = args.data_path, obj=args.obj, num=args.train_num)
    
    save_path = os.path.join(args.save_prefix, args.obj, f'{args.train_num}_{args.sigma}_{args.lam1}_{args.lam2}')
    os.makedirs(save_path, exist_ok=True)
    
    from argparse import Namespace
    USER_DIR = "./av_hubert/avhubert"
    fairseq_utils.import_user_module(Namespace(user_dir=USER_DIR))
    ckpt_path = args.ckpt_path
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    avhu_model = models[0]
    if hasattr(avhu_model, "decoder"):
        print("Checkpoint: fine-tuned")
        avhu_model = avhu_model.encoder.w2v_model
    else:
        print("Checkpoint: pre-trained w/o fine-tuning")
    
    train(args, avhu_model=avhu_model, task=task, dataset=dataset, device=device, save_path=save_path)

    
# CUDA_VISIBLE_DEVICES=3 python new_train3.py --obj ID_0
        
