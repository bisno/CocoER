import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import LambdaLR

from thirdparty.clip import clip
import os, math
from itertools import permutations
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt



def load_clip_to_cpu(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model.float()


def load_clip_to_gpu(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda:1").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda:1")

    model = clip.build_model(state_dict or model.state_dict())
    return model.float()


class txt_loss1(nn.Module):
    def __init__(self):
        super(txt_loss1, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

    def forward(self, logits_image, logits_text):
        batch_size = logits_text.shape[0]
        labels = torch.arange(batch_size).cuda()
        output_image = self.cross_entropy_loss(logits_image, labels)
        output_text = self.cross_entropy_loss(logits_text, labels)
        output = (output_image + output_text) / 2
        return output
    

def test_scikit_ap(logger, cat_preds, cat_labels, ind2cat):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions.
  :param cat_labels: Categorical emotion labels.
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :return: Numpy array containing average precision per emotion category.
  '''
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])

    logger.info ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  logger.info ('Mean AP %.5f' %(ap.mean()))
  return ap.mean()


class DiscreteLoss(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, loss_type='l2', weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss, self).__init__()
        self.loss_type = loss_type
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 26)) / 26.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1435, 0.1870, 0.1692, 0.1165, 0.1949, 0.1204, 0.1728, 0.1372, 0.1620,
                                              0.1540, 0.1987, 0.1057, 0.1482, 0.1192, 0.1590, 0.1929, 0.1158, 0.1907,
                                              0.1345, 0.1307, 0.1665, 0.1698, 0.1797, 0.1657, 0.1520,
                                              0.1537]).unsqueeze(0)
            self.weights = self.weights.to(self.device)
        if loss_type in ['focal', 'balance_focal']:
            self.alpha = 0.25
            self.gamma = 2

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)

        if self.loss_type == 'l2':
            # pred = torch.softmax(pred, dim=1)
            pred = torch.sigmoid(pred)
            loss = (((pred - target) ** 2) * self.weights).sum(dim=-1).mean()
        elif self.loss_type == 'multilabel':
            # pred = torch.sigmoid(pred)
            loss = F.multilabel_soft_margin_loss(pred, target, weight=self.weights, reduction='none').mean()
        elif self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(pred, target, weight=self.weights, reduction='none').sum(dim=-1).mean()
        elif self.loss_type == 'focal':
            pred_sigmoid = torch.sigmoid(pred)
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
            focal_weight = alpha_weight * torch.pow(pt, self.gamma)

            bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)
            loss = focal_weight * bce_loss
            loss = loss.sum(dim=-1).mean()
            # loss = (loss * self.weights).sum(dim=-1).mean()
        elif self.loss_type == 'balance_focal':
            pred_sigmoid = torch.sigmoid(pred)
            # self.alpha = self.dynamic_alpha_weights2(target, 1.8) # NEW ADD
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
            # focal_weight = alpha_weight * torch.pow(pt, self.gamma)
            focal_weight = self.dynamic_alpha_weights4(target, 0.8) * alpha_weight * torch.pow(pt, self.gamma) # NEW ADD

            bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)
            loss = focal_weight * bce_loss
            loss = loss.sum(dim=-1).mean()
            # loss = (loss * self.weights).sum(dim=-1).mean()
        else:
            raise NotImplementedError

        return loss

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 26))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001 # 1.2683 #
        return weights

    def dynamic_alpha_weights(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = torch.clamp(neg_stats / (pos_stats + neg_stats), min=0.01, max=0.99) # [1, 26]
        return torch.pow(alpha, p)

    def dynamic_alpha_weights2(self, target, p):
        pos_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = 1.0 / torch.log(pos_stats + 2.75)

        return torch.pow(alpha, p)

    def dynamic_alpha_weights4(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        alpha = 1.0 / torch.log(pos_stats + 2.75)
        # alpha[pos_stats == 0] = 0.15  # w/o: v5

        return torch.pow(alpha, p)

    def dynamic_alpha_weights6(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        diff_stats = neg_stats - pos_stats

        return torch.exp(diff_stats * p)

    def dynamic_weights(self, target):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = (neg_stats - pos_stats) / (pos_stats + neg_stats)
        # alpha[alpha == 1] = -6
        return alpha



def get_lr_schedule_with_steps_and_warmup(decay_type, optimizer, warmup_steps, drop_steps=None, gamma=None, total_steps=None):
    def lr_lambda(current_step):

        if current_step < warmup_steps:
            alpha = current_step / warmup_steps
            warmup_factor = 0.01 * (1 - alpha) + alpha

            return warmup_factor

        if decay_type == 'constant':
            return 1.0
        elif decay_type == 'linear':
            # return 1.0 * (current_step / total_steps)
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        elif decay_type == 'cosine':
            return 1.0 * (math.cos(((current_step - warmup_steps) / max(1, total_steps - warmup_steps)) * math.pi) + 1) / 2
        elif decay_type == 'milestone':
            return 1.0 * math.pow(gamma, int((current_step - warmup_steps) / drop_steps))
        else:
            raise NotImplementedError

    return LambdaLR(optimizer, lr_lambda)



def confusion():
    all_data = np.load('./trainval_info.npy', allow_pickle=True).item()

    data = []
    for k in all_data.keys():
        seqs = all_data[k]
        data += seqs

    print (len(data))

    union = np.zeros((26, 26))
    inter = 0
    for info in data:
        image_context, image_body, image_head, cat_label, _, body_coord, head_coord = info

        inter += np.array(cat_label)

        idx = np.where(cat_label == 1)[0]
        for comb in permutations(idx, 2):
            union[comb[0], comb[1]] += 1

    inter = inter.reshape(1,-1) + inter.reshape(-1,1) - union
    out = union / inter #  + np.eye(26)
    print (out)
    # print (np.max(out))
    # print (np.min(out))


    clustering = DBSCAN(eps=0.05, min_samples=5, metric='precomputed')
    clustering.fit(out)
    print (clustering.labels_)

    plt.imshow(out, cmap='jet')
    plt.show()



class DiscreteLoss_7(nn.Module):
    ''' Class to measure loss between categorical emotion predictions and labels.'''

    def __init__(self, loss_type='l2', weight_type='mean', device=torch.device('cpu')):
        super(DiscreteLoss_7, self).__init__()
        self.loss_type = loss_type
        self.weight_type = weight_type
        self.device = device
        if self.weight_type == 'mean':
            self.weights = torch.ones((1, 7)) / 7.0
            self.weights = self.weights.to(self.device)
        elif self.weight_type == 'static':
            self.weights = torch.FloatTensor([0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428,]).unsqueeze(0)
            self.weights = self.weights.to(self.device)
        if loss_type in ['focal', 'balance_focal']:
            self.alpha = 0.25
            self.gamma = 2

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, pred, target):
        if self.weight_type == 'dynamic':
            self.weights = self.prepare_dynamic_weights(target)
            self.weights = self.weights.to(self.device)

        if self.loss_type == 'l2':
            # pred = torch.softmax(pred, dim=1)
            pred = torch.sigmoid(pred)
            loss = (((pred - target) ** 2) * self.weights).sum(dim=-1).mean()
        elif self.loss_type == 'multilabel':
            # pred = torch.sigmoid(pred)
            loss = F.multilabel_soft_margin_loss(pred, target, weight=self.weights, reduction='none').mean()
        elif self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(pred, target, weight=self.weights, reduction='none').sum(dim=-1).mean()
        elif self.loss_type == 'focal':
            pred_sigmoid = torch.sigmoid(pred)
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
            focal_weight = alpha_weight * torch.pow(pt, self.gamma)

            bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)
            loss = focal_weight * bce_loss
            loss = loss.sum(dim=-1).mean()
            # loss = (loss * self.weights).sum(dim=-1).mean()
        elif self.loss_type == 'balance_focal':
            pred_sigmoid = torch.sigmoid(pred)
            # self.alpha = self.dynamic_alpha_weights2(target, 1.8) # NEW ADD
            alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
            pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
            # focal_weight = alpha_weight * torch.pow(pt, self.gamma)
            focal_weight = self.dynamic_alpha_weights4(target, 0.8) * alpha_weight * torch.pow(pt, self.gamma) # NEW ADD

            bce_loss = self.sigmoid_cross_entropy_with_logits(pred, target)
            loss = focal_weight * bce_loss
            loss = loss.sum(dim=-1).mean()
            # loss = (loss * self.weights).sum(dim=-1).mean()
        else:
            raise NotImplementedError

        return loss

    def prepare_dynamic_weights(self, target):
        target_stats = torch.sum(target, dim=0).float().unsqueeze(dim=0).cpu()
        weights = torch.zeros((1, 7))
        weights[target_stats != 0] = 1.0 / torch.log(target_stats[target_stats != 0].data + 1.2)
        weights[target_stats == 0] = 0.0001 # 1.2683 #
        return weights

    def dynamic_alpha_weights(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = torch.clamp(neg_stats / (pos_stats + neg_stats), min=0.01, max=0.99) # [1, 26]
        return torch.pow(alpha, p)

    def dynamic_alpha_weights2(self, target, p):
        pos_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = 1.0 / torch.log(pos_stats + 2.75)

        return torch.pow(alpha, p)

    def dynamic_alpha_weights4(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        alpha = 1.0 / torch.log(pos_stats + 2.75)
        # alpha[pos_stats == 0] = 0.15  # w/o: v5

        return torch.pow(alpha, p)

    def dynamic_alpha_weights6(self, target, p):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        diff_stats = neg_stats - pos_stats

        return torch.exp(diff_stats * p)

    def dynamic_weights(self, target):
        pos_stats = torch.sum(target, dim=0, keepdim=True).float() # [1, 26]
        neg_stats = torch.sum(target == 0, dim=0, keepdim=True).float() # [1, 26]
        alpha = (neg_stats - pos_stats) / (pos_stats + neg_stats)
        # alpha[alpha == 1] = -6
        return alpha

