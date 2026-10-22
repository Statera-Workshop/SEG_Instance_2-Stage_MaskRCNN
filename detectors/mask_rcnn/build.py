"""
Mask R-CNN (ResNet-50-FPN) builder - the 2-Stage instance segmentation reference. Seg task.

Paper: He, Gkioxari, Dollar & Girshick, "Mask R-CNN" (ICCV 2017, arXiv 1703.06870).
Code : torchvision 0.20.1 models/detection/mask_rcnn.py (MultiScaleRoIAlign 14x14, MaskRCNNHeads 4 x conv 256,
       MaskRCNNPredictor deconv 2x + 1x1 conv) and roi_heads.maskrcnn_loss, used directly; torchvision is the
       approved official-equivalent source (the authors' code is Detectron / Caffe2). The box part is the
       detection stage's Faster R-CNN (detectors/faster_rcnn), unchanged.

    mask head  MultiScaleRoIAlign 14x14 (P2-P5, sampling 2) -> 4 x (conv 3x3 256 + ReLU) -> deconv 2x 256 + ReLU
               -> 1x1 conv (C + 1) class-specific 28x28 masks, per-pixel BCE on positive RoIs of the matched
               class, targets = ground-truth masks RoI-aligned to the proposal (torchvision maskrcnn_loss)
    inference  mask head on the final (post-NMS) boxes, channel of the predicted class, sigmoid, pasted
               into the box and thresholded at 0.5 (tasks/seg_task.py)

Raw contract: the Faster R-CNN rows ([B, P * C, 5 + C], one row per (proposal, class)); aux = FPN features.

[Unified] same as Faster R-CNN (no ImageNet weights, trainable BN, common init, no resize / mean-std, Adam 1e-4,
common post-processing, PreciseBN); the loss is the plain sum of RPN, box and mask losses (as torchvision).
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import maskrcnn_loss
from torchvision.ops import MultiScaleRoIAlign

from detectors.faster_rcnn.build import FasterRCNNDet


def split_seg_targets(targets, batch_size, img_size):
    """{"boxes": [N, 6], "masks": [N, S, S]} -> per-image dicts (boxes xyxy px, labels 1..C, masks uint8);
    same degenerate-box filter as detectors.anchor_free.common.split_targets."""
    t_all, m_all = targets["boxes"], targets["masks"]
    out = []
    for b in range(batch_size):
        sel = t_all[:, 0] == b
        t, m = t_all[sel], m_all[sel]
        cxcywh = t[:, 2:6] * img_size
        boxes = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 1e-3) & (boxes[:, 3] - boxes[:, 1] > 1e-3)
        out.append({"boxes": boxes[keep], "labels": t[keep, 1].long() + 1, "masks": m[keep]})
    return out


class MaskRCNNDet(FasterRCNNDet):
    def __init__(self, num_classes, img_size=608):
        super().__init__(num_classes, img_size)
        self.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        self.mask_head = MaskRCNNHeads(256, (256, 256, 256, 256), 1)
        self.mask_predictor = MaskRCNNPredictor(256, 256, num_classes + 1)

    def mask_logits(self, features, boxes, image_sizes):
        return self.mask_predictor(self.mask_head(self.mask_roi_pool(features, boxes, image_sizes)))

    def forward(self, x, targets=None):
        B, C = x.shape[0], self.num_classes
        if targets is None and self.training:                 # PreciseBN forward (see FasterRCNNDet)
            self._features(x)
            return x.new_zeros(B, 0, 5 + C)
        if targets is None:
            images, features, _ = self._features(x)
            proposals, _ = self.det.rpn(images, features)
            n = min(len(p) for p in proposals)
            proposals = [p[:n] for p in proposals]
            class_logits, box_regression = self.roi_outputs(features, proposals, images.image_sizes)
            raw = self.to_raw(class_logits, box_regression, proposals, B)
            return raw, {"features": features, "image_sizes": images.image_sizes}

        tgt = split_seg_targets(targets, B, x.shape[2])
        images, features, tgt = self._features(x, tgt)
        proposals, rpn_losses = self.det.rpn(images, features, tgt)
        proposals, matched_idxs, labels, regression_targets = self.det.roi_heads.select_training_samples(proposals, tgt)
        class_logits, box_regression = self.roi_outputs(features, proposals, images.image_sizes)
        loss_classifier, loss_box_reg = self.box_loss(class_logits, box_regression, labels, regression_targets)

        pos = [l > 0 for l in labels]
        pos_props = [p[k] for p, k in zip(proposals, pos)]
        if sum(int(k.sum()) for k in pos):
            logits = self.mask_logits(features, pos_props, images.image_sizes)
            loss_mask = maskrcnn_loss(logits, pos_props, [t["masks"] for t in tgt], [t["labels"] for t in tgt],
                                      [m[k] for m, k in zip(matched_idxs, pos)])
        else:                                                  # keep the graph (no positive RoI in the batch)
            loss_mask = sum(p.sum() for p in self.mask_predictor.parameters()) * 0.0

        loss_dict = {**rpn_losses, "loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg,
                     "loss_mask": loss_mask}
        loss = sum(loss_dict.values())
        lab = torch.cat(labels)
        fg = lab > 0
        self.stats.update({**loss_dict, "losses": loss, "num_pos": fg.float().sum() / B},
                          F.softmax(class_logits.detach(), -1)[:, 1:].max(-1).values,
                          class_logits.detach()[:, 1:], fg, lab[fg] - 1)
        return loss, x.new_zeros(B, 0, 5 + C)

    @torch.no_grad()
    def decode_masks(self, aux, b, rows, boxes, labels):
        feats = OrderedDict((k, v[b:b + 1]) for k, v in aux["features"].items())
        logits = self.mask_logits(feats, [boxes], [aux["image_sizes"][b]])          # [K, C + 1, 28, 28]
        idx = torch.arange(len(labels), device=logits.device)
        return {"kind": "roi", "prob": logits[idx, labels + 1].sigmoid()}


def build_model(task, num_classes):
    return MaskRCNNDet(num_classes)
