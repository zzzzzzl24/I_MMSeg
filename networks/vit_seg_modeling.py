# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

from os.path import join as pjoin
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from collections import OrderedDict
from pathlib import Path



logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config, input, hidden, output):
        super(Mlp, self).__init__()
        self.fc1 = Linear(input, hidden)
        self.fc2 = Linear(hidden, output)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        x = self.act_fn(x)
        x= torch.sigmoid(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.hybrid = True 
        self.config = config
        self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, image):
        if self.hybrid:
            x, features = self.hybrid_model(image)
            x = self.dropout(x)
            return x, features


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config)

    def forward(self, input_ids):
        x, features = self.embeddings(input_ids)
        return x, features

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            head_channels*2,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )#Used to adjust the hidden_size channel count back to head_channels
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])#[512, 256, 128, 64]
        out_channels = decoder_channels # (256, 128, 64, 16)

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels #[512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]#Configure jump connections: whether to use them and how many to employ

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]#Construct decoding
        self.blocks = nn.ModuleList(blocks)

    def forward(self, enc_out, enc_features):
        x = self.conv_more(enc_out).contiguous()#x:(24,512,8,8)
        for i, decoder_block in enumerate(self.blocks):
            if enc_features is not None:
                skip = enc_features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 3, embed_dim, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x_A, x_B, x_C):
        x = torch.concat([x_A, x_B, x_C], dim=1).contiguous()
        x = self.fusion_proj(x)
        x = self.norm(x)
        x = self.activation(x)
        # x = x.flatten(2)
        return x

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, text_embed):
        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        return x
    
class ForwardPredictionHead(nn.Module):
    def __init__(self, config):
        super(ForwardPredictionHead, self).__init__()

        self.config = config
        weight_nums, bias_nums = [], []
        weight_nums.append(4 * 4)#Finally, what is the output channel C of precls_conv? Here it is C*C.
        weight_nums.append(4 * 1)
        bias_nums.append(4)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        
        # Initialize the controller layer
        self.controller = nn.Conv2d(512, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)

        # GAP layers
        self.Gap_64 = nn.Sequential(
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
        )

        self.GAP_256 = nn.Sequential(
            nn.GroupNorm(4, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )

        self.GAP_512 = nn.Sequential(
            nn.GroupNorm(4, 512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
        )

        # Precls convolution layer
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=1)
        )

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_classes = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))  # Split params into weights and biases

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_classes * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_classes, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_classes)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_classes):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features #x:(1, 32, 128, 128)(b,c,h,w),num_classes=4
        for i, (w, b) in enumerate(zip(weights, biases)):#w:(32,8,1,1).(32,8,1,1),(4,8,1,1);b:(32)(32)(4)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_classes
            )
            if i < n_layers - 1:
                x = F.relu(x, inplace=True)
        return x

    def forward(self, feature_list, clip_embedding, decoder_output):
        # Aggregate features from different levels
        feat = (self.Gap_64(feature_list[2]) + self.GAP_256(feature_list[1]) + self.GAP_512(feature_list[0])) / 3
        batch_size = feat.shape[0]
        logits_array = []

        for i in range(batch_size):
            # Combine vision-language embedding
            feat_embed =  feat[i].unsqueeze(0).repeat(self.config.n_classes, 1, 1, 1)
            clip_embed = clip_embedding.unsqueeze(2).unsqueeze(2)
            vision_language_embedding = torch.cat([feat_embed, clip_embed], dim=1)  # [4, 1024, 1, 1]
            
            params = self.controller(vision_language_embedding)
            params.squeeze_(-1).squeeze_(-1)

            # Prepare the decoder output
            decoder_out = decoder_output[i].unsqueeze(0)
            head_inputs = self.precls_conv(decoder_out)

            head_inputs = head_inputs.repeat(self.config.n_classes, 1, 1, 1)  # [4, 8, H, W]

            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # [1, 4*8, H, W]

            # Parse weights and biases
            weights, biases = self.parse_dynamic_params(params, 4, self.weight_nums, self.bias_nums)

            # Apply convolution
            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, H, W))

        # Concatenate logits from all batch items
        out_seg = torch.cat(logits_array, dim=0)

        return out_seg

class get_embedding_list(nn.Module):
    def __init__(self):
        super(get_embedding_list, self).__init__()
        self.features64_to_64 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=1),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=1))

        self.features256_to_128 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1))

        self.features512_to_256 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1))
        
    def forward(self, features_list):
        embedding_list = [
            self.features64_to_64(features_list[2]),
            self.features256_to_128(features_list[1]),
            self.features512_to_256(features_list[0])
        ]
        return embedding_list

def pixelwise_cosine_similarity(A, B, epsilon=1e-8):
    """
    Calculates the cosine similarity between corresponding pixels of two feature tensors.

    Parameters:
        A (torch.Tensor): A feature tensor with shape (batch_size, feature_dim, num_pixels).
        B (torch.Tensor): A feature tensor with shape (batch_size, feature_dim, num_pixels).
        epsilon (float): A small constant to prevent division-by-zero errors, defaulting to 1e-8.

    Returns:
        torch.Tensor: A cosine similarity matrix with shape (batch_size, num_pixels, num_pixels).
    """
    # Verify that the input shapes are consistent
    if A.shape != B.shape:
        raise ValueError("The shapes of input tensors A and B must be identical!")

    # Step 1: Compute the transposed matrix multiplication to obtain the dot product matrix (batch_size, num_pixels, num_pixels)
    dot_product = torch.matmul(A.transpose(1, 2), B)  # A is transposed into (batch_size, num_pixels, feature_dim)

    # Step 2: Calculate the L2 norm for each pixel point (batch_size, num_pixels)
    norm_A = torch.norm(A, p=2, dim=1)
    norm_B = torch.norm(B, p=2, dim=1)

    # Step 3: Compute the outer product of the norm as the denominator matrix (batch_size, num_pixels, num_pixels)
    denominator = torch.matmul(norm_A.unsqueeze(2), norm_B.unsqueeze(1))

    # Step 4: Calculate cosine similarity, guarding against division by zero
    cosine_sim = dot_product / (denominator + epsilon)

    return cosine_sim

def cosine_attention_update(A, B):
    attention_weights = torch.softmax(B, dim=-1)  # (B, N, N)
    
    #  Attention-weighted aggregation
    # --------------------------------------------------
    # Transform feature A into a form suitable for matrix multiplication (B, N, C)
    A_transposed = A.transpose(1, 2)  # (B, N, C)
    
    # Feature reorganisation using attention weights (B, N, N) × (B, N, C) = (B, N, C)
    aggregated_feature = torch.matmul(attention_weights, A_transposed)  # (B, N, C)
    
    # Restore original dimensional order (B, C, N)
    A = aggregated_feature.transpose(1, 2)  # (B, C, N)

    return A

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 8, 8) -> (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 同上
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))  # (B, C)
        max_out = self.mlp(self.max_pool(x).view(b, c))  # (B, C)
        out = avg_out + max_out
        out = self.sigmoid(out).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * out  # Weighted input features

# ----------------- Spatial attention -----------------
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7×7 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average Pooling (B, 1, 8, 8)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Maximum pooling
        out = torch.cat([avg_out, max_out], dim=1)  # (B, 2, 8, 8)
        out = self.sigmoid(self.conv(out))  # (B, 1, 8, 8)
        return x * out  # Weighted input features

# ----------------- CBAM module -----------------
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)  # Perform channel attention first
        x = self.spatial_att(x)  # Proceed with spatial attention
        return x

# ----------------- CBAM + Dimensionality reduction -----------------
class CBAMFeatureReducer(nn.Module):
    def __init__(self, in_channels=1024, reduced_channels=64):
        super(CBAMFeatureReducer, self).__init__()
        self.cbam = CBAM(in_channels)  # CBAM Attention Module
        self.conv1x1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)  # 1×1
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP

    def forward(self, x):
        x = self.cbam(x)  # CBAM 
        x = self.conv1x1(x)  # 1×1 Convolutional Dimension Reduction (B, 1024, 8, 8) -> (B, 256, 8, 8)
        x = self.global_avg_pool(x).view(x.size(0), -1)  # GAP (B, 256, 1, 1) -> (B, 256)
        return x

class Class_Feature_Modulation(nn.Module):
    def __init__(self, text_dim=32):
        super(Class_Feature_Modulation, self).__init__()
        self.fc = nn.Linear(text_dim, 8)  # Directly mapped to four groups of gamma and beta

    def forward(self, x, text_embed):
        """
        x: [4, 256]  Features requiring modulation
        text_embed: [8]  Input text embedding
        """
        params = self.fc(text_embed)  # Output (8,)
        gamma, beta = params.view(4, 2).chunk(2, dim=1)  # Transform into (4, 2), then split into gamma and beta

        x = x * (1 + gamma) + beta  # Linear transformation
        return x



class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes  #Number of categories
        self.zero_head = zero_head #The zero_head parameter is typically used to control the initialisation method for the model's classification head.
        self.classifier = config.classifier  #Define the classifier
        self.patches = config.patches.size
        self.class_embedding = torch.load(f'{PROJECT_ROOT}/text_features/embedding_class_information.pth', weights_only=True).float()
        self.modal_embedding = torch.load(f'{PROJECT_ROOT}/text_features/embedding_MRI_information.pth', weights_only=True).float()
        self.transformer1 = Transformer(config, img_size, vis)
        self.transformer2 = Transformer(config, img_size, vis)
        self.transformer3 = Transformer(config, img_size, vis)
        self.decoder1 = DecoderCup(config)
        self.decoder2 = DecoderCup(config)
        self.decoder3 = DecoderCup(config)
        self.decoder4 = DecoderCup(config)
        self.feature_fusion = nn.Sequential(OrderedDict([
            ('fusion1', Fusion_Embed(512)),
            ('fusion2', Fusion_Embed(256)),
            ('fusion3', Fusion_Embed(64))
        ]))

        self.text_to_64 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True))
        self.text_to_128 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True))
        self.text_to_256 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True))
        self.text_to_vision = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True))
        
        self.GAP_cine = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
            )
        self.GAP_psir = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
        )
        self.GAP_t2w = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
            nn.Flatten(start_dim=1),
        )

        self.modal_text_to_vision = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True))
        
        self.modal_text_to_weight = nn.Sequential(
            Mlp(config, input=512,hidden=128,output=64),
            nn.ReLU(inplace=True))

        self.mix_cine = Fusion_Embed(1024)
        self.mix_psir = Fusion_Embed(1024)
        self.mix_t2w = Fusion_Embed(1024)

        self.CBAMFeatureReducer_cine = CBAMFeatureReducer()
        self.CBAMFeatureReducer_psir = CBAMFeatureReducer()
        self.CBAMFeatureReducer_t2w = CBAMFeatureReducer()
        self.Class_Feature_Modulation_cine = Class_Feature_Modulation()
        self.Class_Feature_Modulation_psir = Class_Feature_Modulation()
        self.Class_Feature_Modulation_t2w = Class_Feature_Modulation()

        self.class_MLP_dec_cine = Mlp(config, input=128,hidden=64,output=32)
        self.class_MLP_dec_psir = Mlp(config, input=128,hidden=64,output=32)
        self.class_MLP_dec_t2w = Mlp(config, input=128,hidden=64,output=32)

        self.fusion_embed = get_embedding_list()
        self.cine_embed = get_embedding_list()
        self.psir_embed = get_embedding_list()
        self.t2w_embed = get_embedding_list()
        self.class_MLP_cine = Mlp(config, input=512,hidden=256,output=64)
        self.class_MLP_psir = Mlp(config, input=512,hidden=256,output=64)
        self.class_MLP_t2w = Mlp(config, input=512,hidden=256,output=64)

        self.forward_prediction_head = ForwardPredictionHead(config)
        self.forward_prediction_head1 = ForwardPredictionHead(config)
        self.forward_prediction_head2 = ForwardPredictionHead(config)
        self.forward_prediction_head3 = ForwardPredictionHead(config)
        self.forward_prediction_head4 = ForwardPredictionHead(config)
        
        self.dec_output_fusion = Fusion_Embed(16)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.segmentation_head1 = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.segmentation_head2 = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.segmentation_head3 = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x, x1, x2, do_contrast=False):
        modal_embedding = self.modal_embedding.cuda()
        modal_features = self.modal_text_to_vision(modal_embedding)
        modal_features_2 = self.modal_text_to_weight(modal_embedding)
        class_embedding = self.class_embedding.cuda()
        class_features = self.text_to_vision(class_embedding)
        if x.size()[1] == 1:
            cine = x.repeat(1,3,1,1)
            psir = x1.repeat(1,3,1,1)
            t2w = x2.repeat(1,3,1,1)
        # encoder
        cine_f, cine_features = self.transformer1(cine)
        psir_f, psir_features = self.transformer2(psir)
        t2w_f, t2w_features = self.transformer3(t2w)
         
        batch_size, C, H, W = cine_f.shape
        modal_feature_cine = torch.cat([modal_features[0].repeat(batch_size,1), self.GAP_cine(cine_f)],dim=1)
        modal_feature_psir = torch.cat([modal_features[1].repeat(batch_size,1), self.GAP_psir(psir_f)],dim=1)
        modal_feature_t2w = torch.cat([modal_features[2].repeat(batch_size,1), self.GAP_t2w(t2w_f)],dim=1) 

        cine_mask = self.class_MLP_cine(modal_feature_cine).reshape(batch_size, H, W).unsqueeze(1).expand(-1, C, -1, -1)
        psir_mask = self.class_MLP_psir(modal_feature_psir).reshape(batch_size, H, W).unsqueeze(1).expand(-1, C, -1, -1) 
        t2w_mask = self.class_MLP_t2w(modal_feature_t2w).reshape(batch_size, H, W).unsqueeze(1).expand(-1, C, -1, -1)

        cine_mask = torch.mul(cine_f, cine_mask).reshape(batch_size,C,-1)
        psir_mask = torch.mul(psir_f, psir_mask).reshape(batch_size,C,-1)
        t2w_mask = torch.mul(t2w_f, t2w_mask).reshape(batch_size,C,-1)

        # Calculate the cosine similarity between each pair of modes
        cine_psir_cos_sim = abs(pixelwise_cosine_similarity(cine_mask, psir_mask))
        cine_t2w_cos_sim = abs(pixelwise_cosine_similarity(cine_mask, t2w_mask))
        psir_cine_cos_sim = abs(pixelwise_cosine_similarity(psir_mask, cine_mask))
        psir_t2w_cos_sim = abs(pixelwise_cosine_similarity(psir_mask, t2w_mask))
        t2w_cine_cos_sim = abs(pixelwise_cosine_similarity(t2w_mask, cine_mask))
        t2w_psir_cos_sim = abs(pixelwise_cosine_similarity(t2w_mask, psir_mask))
 
        cine_cp = cosine_attention_update(cine_mask, cine_psir_cos_sim).reshape(batch_size, C, H, W)
        cine_ct = cosine_attention_update(cine_mask, cine_t2w_cos_sim).reshape(batch_size, C, H, W)
        psir_pc = cosine_attention_update(psir_mask, psir_cine_cos_sim).reshape(batch_size, C, H, W)
        psir_pt = cosine_attention_update(psir_mask, psir_t2w_cos_sim).reshape(batch_size, C, H, W)
        t2w_tc = cosine_attention_update(t2w_mask, t2w_cine_cos_sim).reshape(batch_size, C, H, W)
        t2w_tp = cosine_attention_update(t2w_mask, t2w_psir_cos_sim).reshape(batch_size, C, H, W)

        cine_f = self.mix_cine(cine_f, cine_cp, cine_ct)
        psir_f = self.mix_psir(psir_f, psir_pc, psir_pt)
        t2w_f = self.mix_t2w(t2w_f, t2w_tc, t2w_tp)

        # decoder
        dec_cine_out = self.decoder1(cine_f, cine_features)
        dec_psir_out = self.decoder2(psir_f, psir_features)
        dec_t2w_out = self.decoder3(t2w_f, t2w_features)
        # The resulting feature dimensions are (24, 16, 128, 128).
        
        # Modulating class_features across modalities:
        cine_cbam = torch.mean(self.CBAMFeatureReducer_cine(cine_f),dim=0)
        psir_cbam = torch.mean(self.CBAMFeatureReducer_psir(psir_f),dim=0)
        t2w_cbam = torch.mean(self.CBAMFeatureReducer_t2w(t2w_f),dim=0)
        modal_dec_cine = torch.cat([modal_features_2[0], cine_cbam],dim=0)
        modal_dec_psir = torch.cat([modal_features_2[1], psir_cbam],dim=0)
        modal_dec_t2w = torch.cat([modal_features_2[2], t2w_cbam],dim=0) 
        cine_text = self.class_MLP_dec_cine(modal_dec_cine)
        psir_text = self.class_MLP_dec_psir(modal_dec_psir)
        t2w_text = self.class_MLP_dec_t2w(modal_dec_t2w)
        class_features_cine = self.Class_Feature_Modulation_cine(class_features, cine_text)
        class_features_psir = self.Class_Feature_Modulation_psir(class_features, psir_text)
        class_features_t2w = self.Class_Feature_Modulation_t2w(class_features, t2w_text)

        #Two feature fusion stages: 1. Encoder-stage res feature fusion: resulting in img_features; 2. Fusion of outputs from three encoders: fusion_features.
        img_features_fusion = []
        for i in range(len(psir_features)):
            img_features_fusion.append(self.feature_fusion[i](cine_features[i], psir_features[i], t2w_features[i]))

        #Seg head section
        out = self.dec_output_fusion(dec_cine_out, dec_psir_out, dec_t2w_out)
        out_seg = self.forward_prediction_head(img_features_fusion, class_features, out)

        if do_contrast:
            text_embedding_list = [self.text_to_64(class_embedding),
                                   self.text_to_128(class_embedding),
                                   self.text_to_256(class_embedding)]
            
            cine_embedding_list = self.cine_embed(cine_features)
            psir_embedding_list = self.psir_embed(psir_features)
            t2w_embedding_list =self.t2w_embed(t2w_features)

        else:
            text_embedding_list = None
            cine_embedding_list = None
            psir_embedding_list = None
            t2w_embedding_list = None

        if self.training:
            cine_pre = self.forward_prediction_head1(cine_features, class_features_cine, dec_cine_out)
            psir_pre = self.forward_prediction_head2(psir_features, class_features_psir, dec_psir_out)
            t2w_pre = self.forward_prediction_head3(t2w_features, class_features_t2w, dec_t2w_out)
            return out_seg, (cine_pre, psir_pre, t2w_pre), \
                (cine_embedding_list, psir_embedding_list, t2w_embedding_list),\
                    text_embedding_list
        else:
            return out_seg
    

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}
