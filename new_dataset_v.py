import random, os, csv
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from dataprocess.utils.color_transfer import *
import time
from AugMix_cv2 import AugmentCV
from math import ceil
import torch.nn.functional as F

class AVLips_augmixmix():
    # AVLips
    ID_LIST = [i for i in range(0, 9)]
    ID_MAP = {f"ID_{i}": idx for idx, i in enumerate(ID_LIST)}
    # lombard:
    # users = [f's{i}' for i in (list(range(2,10)) + list(range(16,28)) + list(range(44,56)))]
    # ID_LIST = sorted([u for u in users if int(u[1:]) % 2 == 0], key=lambda x:int(x[1:]))
    # ID_MAP = {u:i for i,u in enumerate(ID_LIST)}
    
    def __init__(self, mode, data_path, obj, augment=True, num=750, aug_severity=5, seg_len=50,
                 mix_prob=True):
        super().__init__()
        assert mode in ['train', 'val', 'others', 'test', 'fake']
        self.mode = mode
        self.obj = obj 
        self.augment = augment
        self.mix_prob = mix_prob
        self.id_map = AVLips_augmixmix.ID_MAP
        self.transform = transforms.Compose([
                            transforms.Resize((96, 96)),   
                            transforms.Grayscale(),
                            transforms.ToTensor()
                        ])
        self.videos, self.fake_pool = self.load_csv(data_path)
        self.augmentor = AugmentCV()
        self.aug_severity = aug_severity
        
        self.video_id_map = {folder: vid for vid, (folder, _label) in enumerate(self.videos)}
        ##
        self.SEG_LEN = seg_len 
        self.samples = []                               # [(folder, label, start_idx), ...]
        for folder, label in self.videos:
            n_frames = len(os.listdir(folder))
            n_seg = n_frames // self.SEG_LEN if n_frames > self.SEG_LEN else 1 
            for k in range(n_seg):
                self.samples.append((folder, label, k * self.SEG_LEN))
        
        self.fake_samples = []
        for folder, label in self.fake_pool:
            n_frames = len(os.listdir(folder))
            n_seg = n_frames // self.SEG_LEN if n_frames > self.SEG_LEN else 1
            for k in range(n_seg):
                self.fake_samples.append((folder, label, k * self.SEG_LEN))
                
        if self.mode in ['train', 'others'] :    
            selected = random.sample(self.samples, k=min(num, len(self.samples)))
            self.samples = selected
                   
    def __len__(self):
        return len(self.samples)

    def to_numpy_gray_frame(self, frame):  # frame: (1, H, W)
        return frame.squeeze(0).numpy().astype(np.float32)  # (H, W)
    
    def to_numpy_frame(self, frame_tensor):
        frame_np = frame_tensor.permute(1, 2, 0).numpy() # (H,W,C)
        return np.repeat(frame_np, 3, axis=2) if frame_np.shape[2] == 1 else frame_np
    
    def __getitem__(self, idx):
        sample, _, st = self.samples[idx]
        real_video = self.load_video(sample, start_idx=st, target_frames=self.SEG_LEN)  # (C, T, H, W)
        C, T, H, W = real_video.shape

        if self.mode == 'train' and self.augment:
            # ---------- SML ----------
            # rand = np.random.rand()
            # if rand > 0.5:
            param_cache = self.augmentor._generate_param_cache(mixture_width=3, aug_severity=self.aug_severity)
            AugFrames = []
            for t in range(T):
                real_frame = self.to_numpy_frame(real_video[:, t, :, :])  # -> HWC
                frame_aug = self.augmentor(real_frame, param_cache=param_cache) / 255.0  # HWC
                frame_aug = frame_aug[:, :, 0] if frame_aug.shape[2] == 3 else frame_aug
                AugFrames.append(torch.from_numpy(frame_aug).unsqueeze(0).float())  # (1, H, W)
            AugVideo = torch.stack(AugFrames, dim=1)  # (1, T, H, W)
                # video_list = [real_video, AugVideo]
                # label_list = [torch.tensor(1), torch.tensor(0)]
                
            # ---------- CML ----------
            # else:
            lam = np.random.beta(1, 1)
            real_list = [(f,l,st) for (f,l,st) in self.samples if f != sample]
            pool = real_list
            breal_sample_path, _, breal_st = random.choice(pool)
            breal_video = self.load_video(breal_sample_path, start_idx=breal_st, target_frames=self.SEG_LEN)
            MixupFrames = []
            for t in range(T):
                real_frame = self.to_numpy_gray_frame(real_video[:, t, :, :])  # (H, W)
                breal_frame = self.to_numpy_gray_frame(breal_video[:, t, :, :])
                mixupframe = lam * real_frame + (1 - lam) * breal_frame
                MixupFrames.append(torch.from_numpy(mixupframe).unsqueeze(0))  # (1, H, W)
            MixupVideo = torch.stack(MixupFrames, dim=1)  # (1, T, H, W)
            # video_list = [real_video, MixupVideo]
            # label_list = [torch.tensor(1), torch.tensor(0)]
            video_list = [real_video, AugVideo, MixupVideo]
            label_list = [torch.tensor(1), torch.tensor(0), torch.tensor(0)]
            
            return video_list, label_list
        else:
            video_list = real_video
            label_list = torch.tensor(0 if self.mode in ['others', 'test', 'fake'] else 1)
            
            video_idx = torch.tensor(self.video_id_map[sample], dtype=torch.long)

            return video_list, label_list, video_idx
    
    def load_csv(self, path, ratio = 0.75):
        with open(path) as f:
            reader = csv.reader(f)
            all_rows = [(row[0], int(row[1])) for row in reader]
        
        label_to_samples = defaultdict(list)
        for video, label in all_rows:
            label_to_samples[label].append((video, label))
        
        videos, fake_pool = [], []
        target_label = self.id_map[self.obj]
        for label, samples in label_to_samples.items():
            split_idx = int(ratio * len(samples)) 
            
            if self.mode == 'train':
                if label == target_label:
                    videos.extend(samples[:split_idx])    
                else:
                    fake_pool.extend(samples[:split_idx]) 
                    
            elif self.mode == 'others':
                if label != target_label:
                    videos.extend(samples[:split_idx])  
            
            elif self.mode == 'val':
                if label == target_label:
                    videos.extend(samples[split_idx:]) 
            
            elif self.mode == 'test':    
                if label != target_label:
                    # videos.extend(random.sample(samples, min(20, len(samples))))
                    videos.extend(random.sample(samples[split_idx:], min(20, len(samples[split_idx:]))))
            
            elif self.mode == 'fake': 
                if label == target_label: # origin fake
                    videos.extend(samples)
        
        return videos, fake_pool
       
    def load_video(self, path, start_idx=0, target_frames=50): # complete path
        all_imgs = sorted(os.listdir(path))
        slice_imgs = all_imgs[start_idx : start_idx + target_frames]
        
        imgs = [self.transform(Image.open(os.path.join(path, img))) for img in slice_imgs] # [C, H, W] 
        if len(imgs) < target_frames:
            imgs.extend([imgs[-1].clone() for _ in range(target_frames - len(imgs))])
        
        video_tensor = torch.stack(imgs[:target_frames]).permute(1, 0, 2, 3) # shape: [T, C, H, W]->[C, T, H, W]
        return video_tensor
    
