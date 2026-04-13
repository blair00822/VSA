'''
video-level metrics
'''
from new_dataset_v import AVLips_augmixmix
from new_train3 import *
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

id_map = AVLips_augmixmix.ID_MAP

if __name__ == "__main__":  
    args = arg_parse()
    set_seed(42)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    # Load AVHuBERT
    from argparse import Namespace
    USER_DIR = "av_hubert/avhubert"
    fairseq_utils.import_user_module(Namespace(user_dir=USER_DIR))
    ckpt_path = args.ckpt_path
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    avhu_model = models[0]
    if hasattr(avhu_model, "decoder"):
        print("Checkpoint: fine-tuned")
        avhu_model = avhu_model.encoder.w2v_model
    else:
        print("Checkpoint: pre-trained w/o fine-tuning")
    
    threshold_dict = {}
    # change this
    checkpoint_path = os.path.join(args.save_prefix, args.obj, f'{args.train_num}_{args.sigma}_{args.lam1}_{args.lam2}', 'checkpoint.pkl')
    preprocess, model = get_model(
            avhubert_model=avhu_model,
            device=device,
            image_crop_size=task.cfg.image_crop_size,
            image_mean=task.cfg.image_mean,
            image_std=task.cfg.image_std,
            backbone_out_dim=args.backbone_out_dim,
            projector_feat_dim=args.projector_feat_dim,
            checkpoint_path=checkpoint_path
        )
    print('Successfully Loaded fc-checkpoint from:', checkpoint_path)
    model.eval()
    # real no aug
    train_data = AVLips_augmixmix('train', data_path = args.data_path, obj=args.obj, augment = False, 
                                    num=args.train_num, seg_len=args.seg_len) 
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_features = []
    with torch.no_grad():
        for videos, *_ in train_loader:
            videos = videos.to(device)
            x = preprocess(videos) 
            feats = model(x)[1].cpu().numpy()
            train_features.append(feats)
    train_features = np.concatenate(train_features, axis=0)
    print("Real features:", train_features.shape)
    
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
    lof.fit(train_features)

    prefix = os.path.dirname(args.data_path)
    paths = [
        args.data_path,
        os.path.join(prefix, args.obj + '_fake.csv'),
    ]
    
    deepfake_name = [args.obj +'real']
    for path in paths:
        filename = os.path.basename(path)  
        basename = filename.split('.')[0]  
        name_before_underscore = basename
        deepfake_name.append(name_before_underscore)
        
    # ================= threshold ==================
    thr_real = AVLips_augmixmix('train', data_path = args.data_path, obj=args.obj, augment=False, 
                                num=args.train_num, seg_len=args.seg_len)
    thr_fake = AVLips_augmixmix('others', data_path = args.data_path, obj=args.obj, augment=False, 
                                num=args.train_num, seg_len=args.seg_len)
    print(len(thr_real), len(thr_fake))
    all_probs_list = []
    for dataset in [thr_real, thr_fake]:
        thr_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        vid2probs = defaultdict(list)
        with torch.no_grad():
            for videos, labels, video_idx in thr_loader:
                videos = videos.to(device)
                x = preprocess(videos) 
                feats = model(x)[1].cpu().numpy()
                v_ids = video_idx.numpy() 
                # video-level
                for f, vid in zip(feats, v_ids):
                    vid2probs[int(vid)].append(f)
        # video-level
        cur_probs = np.stack([
                    np.mean(prob_list, axis=0) 
                    for vid, prob_list in sorted(vid2probs.items(), key=lambda x: x[0])
                ]) 
        all_probs_list.append(cur_probs)
    
    real_probs = all_probs_list[0]
    hm_probs = all_probs_list[1]
    probs_combined = np.concatenate([real_probs, hm_probs])
    labels_combined = np.concatenate([np.zeros(len(real_probs)), np.ones(len(hm_probs))])
    
    scores = -lof.decision_function(probs_combined)
    fpr, tpr, thresholds = roc_curve(labels_combined, scores, drop_intermediate=False)
    index_at_1_fnr = np.where(fpr >= 0.01)[0][0]
    eer_idx = np.nanargmin(np.abs(1 - tpr - fpr))
    threshold_at_1_fnr = thresholds[eer_idx] # or index_at_1_fnr
    threshold_dict[args.obj] = threshold_at_1_fnr
    
    # real test
    test_data = AVLips_augmixmix('val', data_path = args.data_path, obj=args.obj, seg_len=args.seg_len)
    print('real_test:', len(test_data))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # hm + deepfake
    test_loaders = []
    for idx, path in enumerate(paths):
        if idx == 0:
            adver_testdata = AVLips_augmixmix('test', path, obj=args.obj, seg_len=args.seg_len)
        else:
            adver_testdata = AVLips_augmixmix('fake', path, obj=args.obj, seg_len=args.seg_len)
        adver_testloader = DataLoader(adver_testdata, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loaders.append(adver_testloader)

    all_feats_list = []
    all_names_list = []
    # video-level
    all_vfeats_list = []        
    all_vnames_list = [] 

    all_loaders = [test_loader] + test_loaders 
    for name, loader in zip(deepfake_name, all_loaders):
        print(len(loader.dataset))
        cur_feats = []
        # video-level
        vid2feats = defaultdict(list)  # vid -> [feat, feat, ...]
        with torch.no_grad():
            for videos, labels, video_idx in loader:
                videos = videos.to(device)
                x = preprocess(videos) 
                feats = model(x)[1].cpu().numpy()
                v_ids = video_idx.numpy() 
                # clip-level 
                cur_feats.append(feats)
                # video-level
                for f, vid in zip(feats, v_ids):
                    vid2feats[int(vid)].append(f)
        
        # clip-level  
        cur_feats = np.concatenate(cur_feats)
        all_feats_list.append(cur_feats)
        all_names_list.extend([name] * len(cur_feats))
        
        # video-level
        cur_feats_video = np.stack([
                    np.mean(feat_list, axis=0) 
                    for vid, feat_list in sorted(vid2feats.items(), key=lambda x: x[0])
                ]) 
        all_vfeats_list.append(cur_feats_video)
        all_vnames_list.extend([name] * len(cur_feats_video))
        
    tsne_feats = np.concatenate(all_vfeats_list)
    # ================= write to csv file ==================
    csv_dir = 'metrics_results_AD_v4'
    csv_path = os.path.join(csv_dir, f'avlips_{args.train_num}_{args.sigma}_{args.lam1}_{args.lam2}_LOF.csv')
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Level', 'Name', 'Correct', 'Accuracy', 'FRR', 'FAR', 'HTER', 'AUC', 'Recall', 'Precision'])

        # video-level
        real_feats_video = all_vfeats_list[0]
        
        for i in range(1, len(all_vfeats_list)):
            fake_feats_video = all_vfeats_list[i]

            vfeats_combined = np.concatenate([real_feats_video, fake_feats_video], axis=0)
            vlabels_combined = np.concatenate([
                np.zeros(len(real_feats_video), dtype=int),
                np.ones(len(fake_feats_video), dtype=int)
            ], axis=0)

            vscores = -lof.decision_function(vfeats_combined)
            vpreds = (vscores > threshold_dict[args.obj]).astype(int)

            vacc = accuracy_score(vlabels_combined, vpreds)
            vauc = roc_auc_score(vlabels_combined, vscores)
            vrecall = recall_score(vlabels_combined, vpreds)
            vprecision = precision_score(vlabels_combined, vpreds)

            vreal_idx = np.where(vlabels_combined == 0)[0]
            vfake_idx = np.where(vlabels_combined == 1)[0]
            vFRR = (vpreds[vreal_idx] == 1).sum() / len(vreal_idx)
            vFAR = (vpreds[vfake_idx] == 0).sum() / len(vfake_idx)
            vHTER = (vFRR + vFAR)/2

            writer.writerow([
                'video',                
                deepfake_name[i],
                f'{(vpreds == vlabels_combined).sum()}/{len(vpreds)}',
                f'{vacc:.4f}',
                f'{vFRR:.4f}',
                f'{vFAR:.4f}',
                f'{vHTER:.4f}',
                f'{vauc:.4f}',
                f'{vrecall:.4f}',
                f'{vprecision:.4f}'
            ])

        
# CUDA_VISIBLE_DEVICES=2 python new_test_avlips.py --gpu 0 --obj ID_0 