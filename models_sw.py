import torch.nn as nn
import os
import copy
import math
import pickle
import torch.nn.functional as F
import torch
import torchvision.models as torchmodels
from collections import OrderedDict
from future.utils import iteritems
import numpy as np
# from utils import load_clip_to_cpu
from utils import *
from classes import CLASSES, CUSTOM_TEMPLATES
from thirdparty.clip import clip
import logging
import copy

from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
solvers.options['show_progress'] = False


def get_logger(filename, verbosity=1, name=None):
    """logger function for print logs."""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, h, w, x):
        mask = x.new_zeros((x.shape[0], h, w)).bool()
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_length=50, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num_length, num_pos_feats)
        self.col_embed = nn.Embedding(num_length, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, h, w, x):
        # h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def forward_pre(self, x, y, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        x2 = self.self_attn(x2, x2, x2, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        y2 = self.norm3(y)
        x2 = self.cross_attn(x2, y2, y2, mask)
        x = x + self.dropout2(x2)

        x2 = self.norm4(x)
        x2 = self.feed_forward(x2)
        out = x + self.dropout3(x2)

        return out

    def forward_post(self, x, y, mask=None):
        "for conenctions"
        x2 = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        y = self.norm2(y)
        x2 = self.cross_attn(x, y, y, mask)
        x = x + self.dropout2(x2)
        x = self.norm3(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        out = self.norm4(x)

        return out

    def forward(self, x, y, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, mask)
        else:
            return self.forward_post(x, y, mask)


class EncoderLayer2(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(EncoderLayer2, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, y, x_pos, y_pos, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, x_pos)
        x2 = self.self_attn(q, k, x2, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        y2 = self.norm3(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x2, x_pos), self.with_pos_embed(y2, y_pos), y2, mask)
        x = x + self.dropout2(x2)

        x2 = self.norm4(x)
        x2 = self.feed_forward(x2)
        out = x + self.dropout3(x2)

        return out

    def forward_post(self, x, y, x_pos, y_pos,  mask=None):
        "for conenctions"
        q = k = self.with_pos_embed(x, x_pos)
        x2 = self.self_attn(q, k, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        y = self.norm2(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x, x_pos), self.with_pos_embed(y, y_pos), y, mask)
        x = x + self.dropout2(x2)
        x = self.norm3(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        out = self.norm4(x)

        return out

    def forward(self, x, y, x_pos, y_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, x_pos, y_pos, mask)
        else:
            return self.forward_post(x, y, x_pos, y_pos, mask)


class Encoder_selff(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(Encoder_selff, self).__init__()
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, y, x_pos, y_pos, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        y2 = self.norm3(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x2, x_pos), self.with_pos_embed(y2, y_pos), y, mask)
        x = x + self.dropout1(x2)
        return x

    def forward_post(self, x, y, x_pos, y_pos,  mask=None):
        "for conenctions"
        q = k = self.with_pos_embed(x, x_pos)
        x2 = self.cross_attn(q, k, y, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout2(x2)
        out = self.norm2(x)

        return x

    def forward(self, x, y, x_pos, y_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, x_pos, y_pos, mask)
        else:
            return self.forward_post(x, y, x_pos, y_pos, mask)


class Encoder(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(Encoder, self).__init__()
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, y, x_pos, y_pos, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        y2 = self.norm3(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x2, x_pos), self.with_pos_embed(y2, y_pos), y, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        x2 = self.feed_forward(x2)
        out = x + self.dropout2(x2)

        return out

    def forward_post(self, x, y, x_pos, y_pos,  mask=None):
        "for conenctions"
        q = k = self.with_pos_embed(x, x_pos)
        x2 = self.cross_attn(q, k, y, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout2(x2)
        out = self.norm2(x)

        return out

    def forward(self, x, y, x_pos, y_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, x_pos, y_pos, mask)
        else:
            return self.forward_post(x, y, x_pos, y_pos, mask)


class Decoder(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(Decoder, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, y, x_pos, y_pos, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        q = k = self.with_pos_embed(x2, x_pos)
        x2 = self.self_attn(q, k, x2, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm2(x)
        y2 = self.norm4(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x2, x_pos), self.with_pos_embed(y2, y_pos), y, mask)
        x = x + self.dropout2(x2)

        x2 = self.norm3(x)
        x2 = self.feed_forward(x2)
        out = x + self.dropout3(x2)

        return out

    def forward_post(self, x, y, x_pos, y_pos,  mask=None):
        "for conenctions"
        q = k = self.with_pos_embed(x, x_pos)
        x2 = self.self_attn(q, k, x, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.cross_attn(self.with_pos_embed(
            x, x_pos), self.with_pos_embed(y, y_pos), y, mask)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        out = self.norm3(x)

        return out

    def forward(self, x, y, x_pos, y_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, x_pos, y_pos, mask)
        else:
            return self.forward_post(x, y, x_pos, y_pos, mask)


class Decoder2(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, n_head, d_model, d_ff, dropout, norm_before=True):
        super(Decoder2, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.cross_attn = MultiHeadAttention(n_head, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.size = size
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, x, y, x_pos, y_pos, mask=None):
        "for conenctions"
        x2 = self.norm1(x)
        y2 = self.norm2(y)
        x2 = self.cross_attn(self.with_pos_embed(
            x2, x_pos), self.with_pos_embed(y2, y_pos), y, mask)
        x = x + self.dropout1(x2)

        x2 = self.norm3(x)
        q = k = self.with_pos_embed(x2, x_pos)
        x2 = self.self_attn(q, k, x2, mask)
        x = x + self.dropout2(x2)

        x2 = self.norm4(x)
        x2 = self.feed_forward(x2)
        out = x + self.dropout3(x2)

        return out

    def forward_post(self, x, y, x_pos, y_pos,  mask=None):
        "for conenctions"
        x2 = self.cross_attn(self.with_pos_embed(
            x, x_pos), self.with_pos_embed(y, y_pos), y, mask)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        q = k = self.with_pos_embed(x, x_pos)
        x2 = self.self_attn(q, k, x, mask)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        out = self.norm3(x)

        return out

    def forward(self, x, y, x_pos, y_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(x, y, x_pos, y_pos, mask)
        else:
            return self.forward_post(x, y, x_pos, y_pos, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) do all the linear projection in batch from d_model => h * d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, attn_weights = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        # 3) concat using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute Scaled Dot Product Attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = Swish()  # nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def convert_weights(state_dict):
    """convert wights when load model."""
    tmp_weights = OrderedDict()
    for key, params in iteritems(state_dict['model']):
        tmp_weights[key] = params
    return tmp_weights


class VI_module_w(nn.Module):
    def __init__(self, args):
        super(VI_module_w, self).__init__()
        self.args = args
        self.device = f"cuda:{self.args.gpu}"

        ft_clip = ['ln_final', 'text_projection']

        # load_clip_to_cpu('RN50')  clip.load("RN50")
        clip_model, _ = clip.load('RN50', device=self.device)

        self.image_encoder = clip_model.encode_image
        self.text_encoder = clip_model.encode_text

        txt_split = 'Photo'  # 'Photo'  Face, Person
        self.classnames = CLASSES
        self.templates = CUSTOM_TEMPLATES[txt_split]
        self.feature_dim = 1024

        self.W = torch.nn.Linear(
            self.feature_dim, self.feature_dim, bias=False).to(self.device)
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.W.weight)

    @staticmethod
    def set_finetune_parameters(model, ft_layer):
        for n, p in model.named_parameters():
            if any([n.startswith(l) for l in ft_layer]) or ft_layer == 'all':
                p.requires_grad = True
            else:
                p.requires_grad = False

    def forward(self, images_context, labels_cat):

        text_inputs = torch.cat(
            [clip.tokenize(self.templates.format(c)) for c in self.classnames])  # 26 * 77
        text_inputs = text_inputs.to(self.device).detach()

        with torch.no_grad():
            text_features = self.text_encoder(text_inputs)
            text_features = text_features.float()
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            image_features = self.image_encoder(images_context)
            image_features = image_features.float()
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)

        bs = image_features.shape[0]
        mapped_image_features = self.W(image_features)
        # mapped_image_features.expand(len(self.classnames),bs,self.feature_dim)
        target_image_features = torch.transpose(mapped_image_features.expand(
            len(self.classnames), bs, self.feature_dim), 0, 1)

        target_text_features = text_features.expand(
            bs, text_features.shape[0], text_features.shape[1])

        return target_image_features, target_text_features


class model_GWT(nn.Module):
    def __init__(self, args):
        super(model_GWT, self).__init__()
        self.args = args
        h, d_model, d_ff, dropout = 4, 256, 1024, 0.1
        # h, d_model, d_ff, dropout = 8, 512, 2048, 0.1
        self.num_grid = 7
        self.num_grid_gwt = 7  # 2 4 8
        num_query = 16

        ft_layers = 'all'  # ['layer4', 'fc']
        model = torchmodels.resnet50(pretrained=True)
        self.set_finetune_parameters(model, ft_layers)
        model_head = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                   model.layer1, model.layer2, model.layer3, model.layer4)
        self.head_proj = nn.Conv2d(
            model.fc.in_features, d_model, kernel_size=1)

        model = torchmodels.resnet50(pretrained=True)
        self.set_finetune_parameters(model, ft_layers)
        model_body = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                   model.layer1, model.layer2, model.layer3, model.layer4)
        self.body_proj = nn.Conv2d(
            model.fc.in_features, d_model, kernel_size=1)

        model = torchmodels.resnet50(pretrained=True)
        self.set_finetune_parameters(model, ft_layers)
        model_context = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                      model.layer1, model.layer2, model.layer3, model.layer4)
        self.context_proj = nn.Conv2d(
            model.fc.in_features, d_model, kernel_size=1)

        # build layers
        self.backbone = nn.ModuleDict({
            'head': model_head,
            'body': model_body,
            'context': model_context,
        })

        self.en_pos = PositionEmbeddingLearned(7, d_model // 2)
        self.query_embedding = nn.Embedding(num_query, d_model)
        self.encoder_head = clones(Encoder_selff(
            h, d_model, d_ff, dropout, norm_before=True), args.num_block1)
        self.encoder_body = clones(Encoder_selff(
            h, d_model, d_ff, dropout, norm_before=True), args.num_block1)
        self.encoder_ctx = clones(Encoder_selff(
            h, d_model, d_ff, dropout, norm_before=True), args.num_block1)
        self.decoder_head = clones(
            Decoder(h, d_model, d_ff, dropout, norm_before=True), args.num_block2)
        self.decoder_body = clones(
            Decoder(h, d_model, d_ff, dropout, norm_before=True), args.num_block2)
        self.decoder_ctx = clones(
            Decoder(h, d_model, d_ff, dropout, norm_before=True), args.num_block2)

        mid_dim = 2048
        mid_dim_gwt = 256
        self.en_pos_gwt = PositionEmbeddingLearned(
            self.num_grid_gwt, mid_dim_gwt // 2)
        num_block1, num_block2 = 3, 3
        h = 4
        self.query_embedding_gwt = nn.Embedding(num_query, mid_dim_gwt)
        self.encoder1 = clones(
            Encoder(h, mid_dim_gwt, d_ff, dropout, norm_before=True), num_block1)
        self.encoder2 = clones(
            Encoder(h, mid_dim_gwt, d_ff, dropout, norm_before=True), num_block1)
        self.encoder3 = clones(
            Encoder(h, mid_dim_gwt, d_ff, dropout, norm_before=True), num_block1)

        self.num_class = args.num_class
        self.device = torch.device('cuda:{}'.format(args.gpu[0]))

        # 2025_07_05

        self.cls_HFE_ctx = nn.Sequential(nn.GELU(), nn.Linear(
            7*7*d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, args.num_class))
        self.cls_HFE_body = nn.Sequential(nn.GELU(), nn.Linear(
            7*7*d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, args.num_class))
        self.cls_HFE_head = nn.Sequential(nn.GELU(), nn.Linear(
            7*7*d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, args.num_class))

        self.cls_global_all = nn.Sequential(nn.GELU(),
                                            nn.LayerNorm(4*d_model),
                                            nn.Linear(4*d_model, d_model),
                                            nn.GELU(),
                                            nn.LayerNorm(d_model),
                                            nn.Linear(d_model, d_model),
                                            nn.GELU(),
                                            nn.LayerNorm(d_model),
                                            nn.Linear(d_model, args.num_class),
                                            )

        self.loss_func = DiscreteLoss(
            self.args.loss_type, 'dynamic', self.device).to(self.device)
        self.ln_after = nn.LayerNorm(mid_dim_gwt)

        self.shar_after = nn.Linear(mid_dim_gwt * num_query, mid_dim)
        self.cls_en_all = nn.Sequential(nn.GELU(), nn.LayerNorm(
            mid_dim), nn.Linear(mid_dim, args.num_class))

        ft_clip = ['ln_final', 'text_projection']
        clip_model, _ = clip.load('RN50', device=self.device)
        for param in clip_model.parameters():
            param.requires_grad = False
        self.image_encoder = clip_model.encode_image
        self.text_encoder = clip_model.encode_text

        txt_split = 'Photo'
        self.classnames = CLASSES
        self.templates = CUSTOM_TEMPLATES[txt_split]

        # build layers
        self.feature_dim = 1024

        self.W = torch.nn.Linear(
            self.feature_dim, self.feature_dim, bias=False).to(self.device)
        self.W.weight = nn.Parameter(torch.load(
            './checkpoints/VI_weights/w.pth')['model']["W.weight"])
        self.VI_ffn = nn.Sequential(nn.GELU(), nn.Linear(self.feature_dim*args.num_class, self.feature_dim*len(CLASSES) // 4), nn.GELU(
        ), nn.Linear(self.feature_dim*args.num_class//4, mid_dim_gwt), nn.GELU(), nn.Linear(mid_dim_gwt, args.num_class))

    @staticmethod
    def set_finetune_parameters(model, ft_layer):
        for n, p in model.named_parameters():
            if any([n.startswith(l) for l in ft_layer]) or ft_layer == 'all':
                p.requires_grad = True
            else:
                p.requires_grad = False
                # if 'bn' in n or 'batchnorm' in n:
                #     p.track_running_stats = False

    @staticmethod
    def set_finetune_parameters_off(model, ft_layer):
        for n, p in model.named_parameters():
            if any([n.startswith(l) for l in ft_layer]) or ft_layer == 'all':
                p.requires_grad = False
            else:
                p.requires_grad = True
                # if 'bn' in n or 'batchnorm' in n:
                #     p.track_running_stats = False

    @staticmethod
    def set_finetune_parameters_pret_all(self, turn='on'):
        if turn == 'off':
            self.set_finetune_parameters_off(self.backbone['head'], 'all')
            self.set_finetune_parameters_off(self.backbone['body'], 'all')
            self.set_finetune_parameters_off(self.backbone['context'], 'all')
            self.set_finetune_parameters_off(self.context_proj, 'all')
            self.set_finetune_parameters_off(self.body_proj, 'all')
            self.set_finetune_parameters_off(self.head_proj, 'all')
            self.set_finetune_parameters_off(self.query_embedding, 'all')
        if turn == 'on':
            self.set_finetune_parameters(self.backbone['head'], 'all')
            self.set_finetune_parameters(self.backbone['body'], 'all')
            self.set_finetune_parameters(self.backbone['context'], 'all')
            self.set_finetune_parameters(self.context_proj, 'all')
            self.set_finetune_parameters(self.body_proj, 'all')
            self.set_finetune_parameters(self.head_proj, 'all')
            self.set_finetune_parameters(self.query_embedding, 'all')

    @staticmethod
    def bilinear_interpolate(im, x, y):
        """
        Args:
            im: (H, W, C) [y, x]
            x: (N)
            y: (N)

        Returns:

        """
        x0 = torch.floor(x).long()
        x1 = x0 + 1

        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
        wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
        wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
        wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
        ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + \
            torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
        return ans

    def get_roi_points(self, rois, grid_size):
        """
        Args:
            rois: [B, 4] (x1y1x2y2)
        """
        local_grid_points = self.get_dense_grid_points(
            rois[:, [3, 2]] - rois[:, [1, 0]], rois.shape[0], grid_size)  # [B, 7*7, yx]
        center_points = (rois[:, [3, 2]] + rois[:, [1, 0]]) / 2.  # [B, yx]
        local_grid_points += center_points.unsqueeze(dim=1)

        return local_grid_points  # [B, 7*7, yx]

    @staticmethod
    def get_dense_grid_points(rois, batch_size, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size))  # [n, n]
        dense_idx = torch.nonzero(faked_features, as_tuple=False)[
            None]  # (N, 2) [y_idx, x_idx]
        dense_idx = dense_idx.repeat(batch_size, 1, 1).float()  # (B, 7x7, 2)

        grid_points = (dense_idx + 0.5) / grid_size * \
            rois.unsqueeze(dim=1) - (rois.unsqueeze(dim=1) / 2)  # (B, 7x7, 2)
        return grid_points

    def vocabulary_informed_output(self, img):

        image_features = self.image_encoder(img)
        image_features = image_features.float()
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)

        mapped_features = self.W(image_features)
        mapped_features = torch.unsqueeze(mapped_features, 1)
        mapped_features = mapped_features.expand(-1, 26, -1).clone()
        mapped_features_flatten = mapped_features.flatten(start_dim=1)

        out_VI = self.VI_ffn(mapped_features_flatten)
        out_VI_feature = self.VI_ffn[:-1](mapped_features_flatten)

        return out_VI, out_VI_feature

    def forward(self, images_context, images_body, images_head, coord_body, coord_head, labels_cat):

        ret = {}

        bs = images_body.shape[0]
        x_head = self.head_proj(
            self.backbone['head'](images_head))  # [B, C, H, W]
        x_body = self.body_proj(
            self.backbone['body'](images_body))  # [B, C, H, W]
        x_context = self.context_proj(
            self.backbone['context'](images_context))  # [B, C, H, W]

        x_head = x_head.flatten(start_dim=2).transpose(
            1, 2)  # [B, HW, C] [32, 49, 256]
        x_body = x_body.flatten(start_dim=2).transpose(
            1, 2)  # [B, HW, C] same as x_head
        x_context = x_context.flatten(start_dim=2).transpose(
            1, 2)  # [B, HW, C] same as x_head

        query_pos = self.query_embedding.weight[None].repeat(bs, 1, 1)
        context_pos = self.en_pos(self.num_grid, self.num_grid, x_context).permute(
            0, 2, 3, 1)  # [B, H, W, C]
        body_pos = self.get_roi_points(
            coord_body, grid_size=self.num_grid) / 32  # [B, N, 2]
        head_pos = self.get_roi_points(
            coord_head, grid_size=self.num_grid) / 32  # [B, N, 2]

        head_x = head_pos[..., 1].view(-1)  # [BN]
        head_y = head_pos[..., 0].view(-1)  # [BN]
        head_pos = self.bilinear_interpolate(context_pos[0], head_x, head_y).view(
            bs, self.num_grid ** 2, -1)  # [B, N, C]
        body_x = body_pos[..., 1].view(-1)  # [BN]
        body_y = body_pos[..., 0].view(-1)  # [BN]
        body_pos = self.bilinear_interpolate(context_pos[0], body_x, body_y).view(
            bs, self.num_grid ** 2, -1)  # [B, N, C]
        context_pos = context_pos.view(bs, self.num_grid ** 2, -1)

        # 2025-07-18
        cls_HFE_head_original = copy.deepcopy(
            dict(self.cls_HFE_head.named_parameters()))
        cls_HFE_body_original = copy.deepcopy(
            dict(self.cls_HFE_body.named_parameters()))
        cls_HFE_ctx_original = copy.deepcopy(
            dict(self.cls_HFE_ctx.named_parameters()))

        for block in self.encoder1:
            # [B, HW, C] [B, 49, 256]
            x_head = block(x_head, x_context, head_pos, context_pos)
        for block in self.encoder2:
            # [B, HW, C] [B, 49, 256]
            x_body = block(x_body, x_context, body_pos, context_pos)
        for block in self.encoder3:
            x_context = block(x_context, x_context, context_pos,
                              context_pos)  # [B, HW, C] [B, 49, 256]

        x_head = self.ln_after(x_head)
        x_body = self.ln_after(x_body)
        x_context = self.ln_after(x_context)
        ret_emo_process_all = []

        # label_VI = []
        VI_class = []

        loss_head_all = []
        loss_body_all = []
        loss_ctx_all = []

        dim = x_head.shape[-1]

        x_head_f_total = x_head.flatten(start_dim=1)  # [B, 49 * 256]
        x_body_f_total = x_body.flatten(start_dim=1)
        x_ctx_f_total = x_context.flatten(start_dim=1)

        label_head_total = self.cls_HFE_head(x_head_f_total)  # [B, 26]
        # [B, 256]
        x_head_feature_total = self.cls_HFE_head[:-1](x_head_f_total)
        label_body_total = self.cls_HFE_body(x_body_f_total)  # [B, 26]
        # [B, 256]
        x_body_feature_total = self.cls_HFE_body[:-1](x_body_f_total)
        label_ctx_total = self.cls_HFE_ctx(x_ctx_f_total)  # [B, 26]
        x_ctx_feature_total = self.cls_HFE_ctx[:-1](x_ctx_f_total)  # [B, 256]

        # create temp_feature to store the refined feature
        # torch.zeros(bs, dim, requires_grad=False).cuda(device=int(self.args.gpu)) # self.cls_HFE_head[:-1](x_head_f) # [B, 256]
        x_head_feature_refined = x_head_feature_total.clone()
        # torch.zeros(bs, dim, requires_grad=False).cuda(device=int(self.args.gpu)) # self.cls_HFE_body[:-1](x_body_f) # [B, 256]
        x_body_feature_refined = x_body_feature_total.clone()
        # torch.zeros(bs, dim, requires_grad=False).cuda(device=int(self.args.gpu)) # self.cls_HFE_ctx[:-1](x_ctx_f) # [B, 256]
        x_ctx_feature_refined = x_ctx_feature_total.clone()

        VI_out_all, VI_out_feature_all = self.vocabulary_informed_output(
            images_context)

        ret['label_VI'] = VI_out_all
        ret['label_head'] = label_head_total
        ret['label_body'] = label_body_total
        ret['label_ctx'] = label_ctx_total

        # take the 1, range from batch_size (bs)
        for bsn in range(bs):

            # x_head.flatten(start_dim=1) # [B, 49 * 256]
            x_head_f = x_head_f_total[bsn].unsqueeze(0)
            label_head = label_head_total[bsn].unsqueeze(0)  # self.cls_HFE_head(x_head_f) # [B, 26]

            x_body_f = x_body_f_total[bsn].unsqueeze(0)  # x_body.flatten(start_dim=1)
            label_body = label_body_total[bsn].unsqueeze(0)  # self.cls_HFE_body(x_body_f)

            x_ctx_f = x_ctx_f_total[bsn].unsqueeze(0)  # x_context.flatten(start_dim=1)
            label_ctx = label_ctx_total[bsn].unsqueeze(0)  # self.cls_HFE_ctx(x_ctx_f)

            VI_out = VI_out_all[bsn].unsqueeze(0)  # [B, n, c], [B, n]

            # calculate loss outputs
            loss_li = []
            weight_grad_li = []
            bias_grad_li = []

            inside_lr = self.args.inside_lr

            with torch.set_grad_enabled(True):

                self.cls_HFE_head.requires_grad_(True)
                self.cls_HFE_body.requires_grad_(True)
                self.cls_HFE_ctx.requires_grad_(True)

                # HFE outputs
                VI_cls = nn.Sigmoid()(VI_out)
                a_head = torch.ones(VI_cls.shape) * 0.3
                a_head = a_head.cuda(device=int(self.args.gpu))
                VI_cls = torch.gt(VI_cls, a_head).type(torch.float32)

                VI_class.append(VI_cls.cpu().detach().numpy())

                criterion_head = self.loss_func
                criterion_head.requires_grad = True
                criterion_head.requires_grad_(True)

                loss_head = criterion_head(label_head, VI_cls)
                loss_head.requires_grad_(True)
                loss_li.append(loss_head)
                loss_head.backward(retain_graph=True)
                weight_grad_li.append(
                    self.cls_HFE_head[-1].weight.grad.cpu().detach().numpy())
                bias_grad_li.append(
                    self.cls_HFE_head[-1].bias.grad.cpu().detach().numpy())

                criterion_body = self.loss_func
                criterion_body.requires_grad = True
                loss_body = criterion_body(label_body, VI_cls)
                loss_li.append(loss_body)
                loss_body.requires_grad_(True)
                loss_body.backward(retain_graph=True)
                weight_grad_li.append(
                    self.cls_HFE_body[-1].weight.grad.cpu().detach().numpy())
                bias_grad_li.append(
                    self.cls_HFE_body[-1].bias.grad.cpu().detach().numpy())

                criterion_ctx = self.loss_func
                criterion_ctx.requires_grad = True
                loss_ctx = criterion_ctx(label_ctx, VI_cls)
                loss_li.append(loss_ctx)
                loss_ctx.requires_grad_(True)
                loss_ctx.backward(retain_graph=True)
                weight_grad_li.append(
                    self.cls_HFE_ctx[-1].weight.grad.cpu().detach().numpy())
                bias_grad_li.append(
                    self.cls_HFE_ctx[-1].bias.grad.cpu().detach().numpy())

                loss_head_all.append(loss_head.item())
                loss_body_all.append(loss_body.item())
                loss_ctx_all.append(loss_ctx.item())

                grad_out_all_step1, lambda_ = cal_grad(
                    grad_list=weight_grad_li, cost_list=loss_li, m=256*26, size_in=256, size_out=26)
                grad_out_all_step1 = torch.from_numpy(
                    grad_out_all_step1).type(torch.float32)
                grad_gwt_norm_all_step1 = torch.nn.functional.normalize(
                    grad_out_all_step1, dim=1)
                grad_mean_all_step1 = torch.mean(
                    grad_gwt_norm_all_step1, 0).cuda(device=int(self.args.gpu))

                bias_grad_out_all_step1, lambda_ = cal_grad(
                    grad_list=bias_grad_li, cost_list=loss_li, m=1*26, size_in=1, size_out=26)
                bias_grad_out_all_step1 = torch.from_numpy(
                    bias_grad_out_all_step1).type(torch.float32)
                bias_grad_gwt_norm_all_step1 = torch.nn.functional.normalize(
                    bias_grad_out_all_step1, dim=1)
                bias_grad_mean_all_step1 = torch.mean(
                    bias_grad_gwt_norm_all_step1, 0).cuda(device=int(self.args.gpu))

                ret_emo_process = []
                # HFE_dict = {0:"head-level", 1:"body-level", 2:"context-level"}
                HFE_dict = ["head-level", "body-level", "context-level"]
                max_index = loss_li.index(max(loss_li))
                ret_emo_process.append(HFE_dict[max_index])
                HFE_dict.pop(max_index)
                loss_li.pop(max_index)
                weight_grad_li.pop(max_index)
                bias_grad_li.pop(max_index)

                # step 2

                grad_out_all_step2, lambda_ = cal_grad(
                    grad_list=weight_grad_li, cost_list=loss_li, m=256*26, size_in=256, size_out=26)
                grad_out_all_step2 = torch.from_numpy(
                    grad_out_all_step2).type(torch.float32)
                grad_gwt_norm_all_step2 = torch.nn.functional.normalize(
                    grad_out_all_step2, dim=1)
                grad_mean_all_step2 = torch.mean(
                    grad_gwt_norm_all_step2, 0).cuda(device=int(self.args.gpu))

                bias_grad_out_all_step2, lambda_ = cal_grad(
                    grad_list=bias_grad_li, cost_list=loss_li, m=1*26, size_in=1, size_out=26)
                bias_grad_out_all_step2 = torch.from_numpy(
                    bias_grad_out_all_step2).type(torch.float32)
                bias_grad_gwt_norm_all_step2 = torch.nn.functional.normalize(
                    bias_grad_out_all_step2, dim=1)
                bias_grad_mean_all_step2 = torch.mean(
                    bias_grad_gwt_norm_all_step2, 0).cuda(device=int(self.args.gpu))

                max_index = loss_li.index(max(loss_li))
                ret_emo_process.append(HFE_dict[max_index])
                HFE_dict.pop(max_index)
                ret_emo_process.append(HFE_dict[0])

                ret_emo_process_all.append(ret_emo_process)

                grad_final = grad_mean_all_step1 + grad_mean_all_step2
                bias_grad_final = bias_grad_mean_all_step1 + bias_grad_mean_all_step2
                grad_distance_all = grad_final.norm() + bias_grad_final.norm()

                self.cls_HFE_head.zero_grad()
                self.cls_HFE_body.zero_grad()
                self.cls_HFE_ctx.zero_grad()

                # print("weight_before: ", torch.max(self.cls_HFE_head[-1].weight.data ), torch.max(self.cls_HFE_head[1].weight.data))

                self.cls_HFE_head[-1].weight.data = self.cls_HFE_head[-1].weight.data - \
                    grad_final * inside_lr
                self.cls_HFE_body[-1].weight.data = self.cls_HFE_body[-1].weight.data - \
                    grad_final * inside_lr
                self.cls_HFE_ctx[-1].weight.data = self.cls_HFE_ctx[-1].weight.data - \
                    grad_final * inside_lr

                self.cls_HFE_head[-1].bias.data = self.cls_HFE_head[-1].bias.data - \
                    bias_grad_final * inside_lr
                self.cls_HFE_body[-1].bias.data = self.cls_HFE_body[-1].bias.data - \
                    bias_grad_final * inside_lr
                self.cls_HFE_ctx[-1].bias.data = self.cls_HFE_ctx[-1].bias.data - \
                    bias_grad_final * inside_lr

                loss_head.backward(retain_graph=True)
                loss_body.backward(retain_graph=True)
                loss_ctx.backward(retain_graph=True)

                self.cls_HFE_head[1].weight.data = self.cls_HFE_head[1].weight.data - \
                    self.cls_HFE_head[1].weight.grad * inside_lr
                self.cls_HFE_body[1].weight.data = self.cls_HFE_body[1].weight.data - \
                    self.cls_HFE_body[1].weight.grad * inside_lr
                self.cls_HFE_ctx[1].weight.data = self.cls_HFE_ctx[1].weight.data - \
                    self.cls_HFE_ctx[1].weight.grad * inside_lr

                self.cls_HFE_head[1].bias.data = self.cls_HFE_head[1].bias.data - \
                    self.cls_HFE_head[1].bias.grad * inside_lr
                self.cls_HFE_body[1].bias.data = self.cls_HFE_body[1].bias.data - \
                    self.cls_HFE_body[1].bias.grad * inside_lr
                self.cls_HFE_ctx[1].bias.data = self.cls_HFE_ctx[1].bias.data - \
                    self.cls_HFE_ctx[1].bias.grad * inside_lr

                # obtain refined feature:

                # [B, 256]
                x_head_feature_refined[bsn] = self.cls_HFE_head[:-
                                                                1](x_head_f)[0]
                # [B, 256]
                x_body_feature_refined[bsn] = self.cls_HFE_body[:-
                                                                1](x_body_f)[0]
                # [B, 256]
                x_ctx_feature_refined[bsn] = self.cls_HFE_ctx[:-1](x_ctx_f)[0]

            # zero grad
            # reset HFE weight
            for name, param in self.cls_HFE_head.named_parameters():
                if name in cls_HFE_head_original:
                    param.data.copy_(cls_HFE_head_original[name].data)

            for name, param in self.cls_HFE_body.named_parameters():
                if name in cls_HFE_body_original:
                    param.data.copy_(cls_HFE_body_original[name].data)

            for name, param in self.cls_HFE_ctx.named_parameters():
                if name in cls_HFE_ctx_original:
                    param.data.copy_(cls_HFE_ctx_original[name].data)

            self.cls_HFE_head.zero_grad()
            self.cls_HFE_body.zero_grad()
            self.cls_HFE_ctx.zero_grad()

        # gather the GWT output

        ret['ret_emo_process_all'] = ret_emo_process_all
        ret['VI_cls'] = VI_class
        ret['loss_head'] = loss_head_all
        ret['loss_body'] = loss_body_all
        ret['loss_ctx'] = loss_ctx_all

        G_feature = torch.cat((x_head_feature_refined, x_body_feature_refined,
                              x_ctx_feature_refined, VI_out_feature_all), dim=1)  # [B, 4 * 256]
        ret['G_feature'] = G_feature

        out_all = self.cls_global_all(G_feature)
        ret['out_all'] = out_all

        self.cls_HFE_head.zero_grad()
        self.cls_HFE_body.zero_grad()
        self.cls_HFE_ctx.zero_grad()
        self.encoder1.zero_grad()
        self.encoder2.zero_grad()
        self.encoder3.zero_grad()
        self.VI_ffn.zero_grad()

        return out_all, grad_distance_all, ret


def get_Gh(grad_list, cost_list_, m):
    cost_list = [cost_list_[i].cpu().detach().numpy()
                 for i in range(len(cost_list_))]

    N = len(cost_list)
    G = np.zeros([N, m])
    b = []

    for i in range(N):
        g = grad_list[i].flatten()
        G[i][:] = g
        b.append(float(cost_list[i]))  # add cost

    b = np.array(b)
    GG = matrix(G)
    hh = matrix(b)

    return GG, hh


def cal_grad(grad_list, cost_list, m, size_in, size_out):

    N = len(cost_list)
    GG, hh = get_Gh(grad_list, cost_list, m)
    P = matrix(GG)*matrix(GG).T
    q = -matrix(hh)
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(np.ones([1, N]))
    b = matrix(np.ones([1]))
    res = qp(P, q, G=G, h=h, A=A, b=b)
    d = -np.array(GG).T.dot(np.array(res['x'])
                            )[:, 0].reshape(size_out, size_in)
    return d, np.array(res['x'])


class TextEncoder2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        # self.logit_scale = clip_model.logit_scale
        # self.text_projection = clip_model.text_projection # [512, 1024]
        # self.ln_final = clip_model.ln_final

        self.text_projection = nn.Parameter(torch.empty(512, 128))
        # self.text_projection = nn.Embedding(512, 128)
        # nn.init.normal_(self.text_projection, std=10)
        # nn.init.kaiming_normal_(self.text_projection, a=0, mode='fan_out')
        self.ln_final = nn.LayerNorm(512)  # x4: 640, x16: 768
        # self.encodeimage = clip_model.encode_image

    def forward(self, text):
        with torch.no_grad():
            x = self.token_embedding(text).type(
                self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # after cuda, x.float()
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #  x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = x @ self.text_projection  # after cuda, x.float()

        return x


class TextEncoder3(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        # self.logit_scale = clip_model.logit_scale
        # self.text_projection = clip_model.text_projection # [512, 1024]
        # self.ln_final = clip_model.ln_final

        self.text_projection = nn.Parameter(torch.empty(512, 128))
        # self.text_projection = nn.Embedding(512, 128)
        # nn.init.normal_(self.text_projection, std=10)
        # nn.init.kaiming_normal_(self.text_projection, a=0, mode='fan_out')
        self.ln_final = nn.LayerNorm(512)  # x4: 640, x16: 768
        self.encodeimage = clip_model.encode_image  # out [1,1024]

    def forward(self, text, image):
        with torch.no_grad():
            x = self.token_embedding(text).type(
                self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            y = self.encodeimage(image).float()
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
              ] @ self.text_projection

        return x, y

# if __name__ == '__main__':
#     model = Res_Emotic_v9([])
