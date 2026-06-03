# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

        # POINT 5: FREEZING FOR FINE-TUNING
        for param in self.network.encoder.parameters():
            param.requires_grad = False

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            # POINT 4.2: FAIR EVALUATION PIPELINE
            # The 12 shared classes (Cityscapes IDs)
            SHARED_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            
            # Mapping: ID_COCO : ID_CITYSCAPES 
            coco_to_city_map = {
                0: 100,   # road -> road
                1: 123,   # sidewalk -> pavement-merged
                2: 129,   # building -> building-other-merged
                3: 131,   # wall -> wall-other-merged
                4: 117,   # fence -> fence-merged
                6: 9,     # traffic light -> traffic light
                8: 116,   # vegetation -> tree-merged
                9: 125,   # terrain -> grass-merged
                10: 119,  # sky -> sky-other-merged
                11: 0,    # person -> person
                13: 2,    # car -> car
                14: 7,    # truck -> truck
                15: 5,    # bus -> bus
                16: 6,    # train -> train
                17: 3,    # motorcycle -> motorcycle
                18: 1     # bicycle -> bicycle
            }

            metric_targets = []
            metric_logits = []

            for b in range(len(targets)):
                # Target Masking (Ground Truth)
                target_b_metric = targets[b].clone()
                mask = torch.ones_like(target_b_metric, dtype=torch.bool)
                for c in SHARED_CLASSES:
                    mask &= (target_b_metric != c)
                
                mask &= (target_b_metric != self.ignore_idx)
                target_b_metric[mask] = self.ignore_idx
                metric_targets.append(target_b_metric)

                # Logit Remapping and Filtering
                logit_b = logits[b]
                if logit_b.shape[0] > 19:
                    # COCO --> 133 classes
                    mapped_logits = torch.full((19, *logit_b.shape[1:]), -1000.0, device=logit_b.device, dtype=logit_b.dtype)
                    for city_id, coco_id in coco_to_city_map.items():
                        mapped_logits[city_id] = logit_b[coco_id]
                    metric_logits.append(mapped_logits)
                else:
                    # Cityscapes --> reset the channels of the non-shared classes
                    mapped_logits = torch.full_like(logit_b, -1000.0)
                    for c in SHARED_CLASSES:
                        mapped_logits[c] = logit_b[c]
                    metric_logits.append(mapped_logits)

            self.update_metrics_semantic(metric_logits, metric_targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
