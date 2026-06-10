import os
import cv2
import glob
import torch
import math
import random
from PIL import Image
import numpy as np
import os.path as osp
import torch.nn.functional as F
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import sys
import yaml

sys.path.append('/content/ML_RL/eomt') 

from models.vit import ViT
from models.eomt import EoMT

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((560, 1120), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((560, 1120), Image.NEAREST),
    ]
)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; or a single glob pattern"
    )  
    parser.add_argument('--loadDir',default="")
    parser.add_argument('--loadWeights', default="")
    parser.add_argument('--subset', default="val")  
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--num_q', type=int, default=200)
    parser.add_argument('--method', type=str, default='max_logit', choices=['max_logit', 'msp', 'max_entropy', 'rba'])
    parser.add_argument(
        '--config', 
        default="/content/ML_RL/eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
    )
    parser.add_argument(
        '--save-logits', 
        action='store_true'
    )
    parser.add_argument(
        '--model_name', type=str, default='eomt'
    )
    args = parser.parse_args()
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    weightspath = args.loadWeights if os.path.isabs(args.loadWeights) else args.loadDir + args.loadWeights

    print ("Loading config: " + args.config)
    print ("Loading weights: " + weightspath)

    with open(args.config, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
        
    vit_backbone_name = config['model']['init_args']['network']['init_args']['encoder']['init_args']['backbone_name']
    num_q = args.num_q
    num_blocks = config['model']['init_args']['network']['init_args']['num_blocks']
    
    num_classes = 133 if args.model_name == 'coco' else 19

    print(f"DINOv2 Encoder Initialization: {vit_backbone_name}...")
    img_size = (560, 1120)
    encoder = ViT(backbone_name=vit_backbone_name, img_size=img_size)

    print(f"EoMT Initialization (Classes: {num_classes}, Queries: {num_q}, Blocks: {num_blocks})...")
    model = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks
    )

    if not args.cpu:
        model = model.cuda()

    state_dict_raw = torch.load(weightspath, map_location='cuda' if not args.cpu else 'cpu')
    
    if 'state_dict' in state_dict_raw:
        state_dict_raw = state_dict_raw['state_dict']
        
    clean_state_dict = {}
    for key, value in state_dict_raw.items():
        clean_key = key.replace('network.', '').replace('model.', '')
        clean_state_dict[clean_key] = value

    if 'encoder.backbone.pos_embed' in clean_state_dict:
        ckpt_pos = clean_state_dict['encoder.backbone.pos_embed']
        model_pos = model.encoder.backbone.pos_embed
        
        if ckpt_pos.shape != model_pos.shape:
            print(f"Resizing pos_embed from {ckpt_pos.shape} to {model_pos.shape}...")

            num_patches_ckpt = ckpt_pos.shape[1] 
            grid_ckpt = int(math.sqrt(num_patches_ckpt))
            
            H_grid, W_grid = model.encoder.backbone.patch_embed.grid_size

            ckpt_pos_2d = ckpt_pos.reshape(1, grid_ckpt, grid_ckpt, -1).permute(0, 3, 1, 2)

            ckpt_pos_resized_2d = F.interpolate(
                ckpt_pos_2d, 
                size=(H_grid, W_grid), 
                mode='bicubic', 
                align_corners=False
            )

            ckpt_pos_resized = ckpt_pos_resized_2d.permute(0, 2, 3, 1).reshape(1, H_grid * W_grid, -1)
            
            clean_state_dict['encoder.backbone.pos_embed'] = ckpt_pos_resized

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"Missing keys: {len(missing_keys)}")

    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            result = model(images)

            pred_masks = result[0][-1][0]
            pred_logits = result[1][-1][0]
            
            prob_queries = torch.nn.functional.softmax(pred_logits, dim=-1)
            known_probs = prob_queries[:, :-1]
            mask_probs = pred_masks.sigmoid()
            
            logits = torch.einsum("qc,qhw->chw", known_probs, mask_probs)
            
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=(560, 1120), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

            if args.method == 'max_logit':
                max_logits = torch.max(logits, dim=0)[0]
                anomaly_result = - max_logits.data.cpu().numpy()
                
            elif args.method == 'msp':
                max_probs = torch.max(logits, dim=0)[0]
                anomaly_result = 1.0 - max_probs.data.cpu().numpy()
                
            elif args.method == 'max_entropy':
                probs = torch.nn.functional.softmax(logits, dim=0)
                log_probs = torch.nn.functional.log_softmax(logits, dim=0)
                entropy = -torch.sum(probs * log_probs, dim=0)
                anomaly_result = entropy.data.cpu().numpy()

            elif args.method == 'rba':
                anomaly_result = -torch.sum(logits, dim=0).data.cpu().numpy()
                        
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
            if args.save_logits:
                cartella_salvataggio = f"saved_logits/saved_logits_{args.model_name}_{args.method}"
                os.makedirs(cartella_salvataggio, exist_ok=True)
                filename = os.path.basename(path).split('.')[0]
                
                save_data = {
                    'pred_logits': logits.cpu(),
                    'ood_gts': ood_gts
                }
                torch.save(save_data, f"{cartella_salvataggio}/{filename}.pt")

            ood_gts_list.append(ood_gts)
            anomaly_score_list.append(anomaly_result)
            
        del result, anomaly_result, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'AUPRC score: {prc_auc*100.0}')
    print(f'FPR@TPR95: {fpr*100.0}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()