import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm

class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        self.lin1 = nn.Linear(embed_features, num_features)
        self.lin2 = nn.Linear(embed_features, num_features)

    def forward(self, inputs, embeds):
        gamma = self.lin1(embeds) # TODO 
        bias = self.lin2(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        self.batchnorm = batchnorm
        self.upsample = upsample
        self.downsample = downsample

        if self.upsample:
            self.up = nn.UpsamplingNearest2d(scale_factor=2)

        if self.downsample:
            self.down = nn.AvgPool2d(kernel_size=2)

        self.skip_connection = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

        if self.batchnorm:
            self.ad_norm1 = AdaptiveBatchNorm(in_channels, embed_channels)
        else:
            self.ad_norm1 = nn.Identity()
        self.relu1 = nn.ReLU()
        self.sp_norm_conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        
        if self.batchnorm:
            self.ad_norm2 = AdaptiveBatchNorm(out_channels, embed_channels)
        else:
            self.ad_norm2 = nn.Identity()        
        self.relu2 = nn.ReLU()
        self.sp_norm_conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        # TODO
        if self.upsample:
            inputs = self.up(inputs)

        if self.batchnorm: 
            outputs = self.ad_norm1(inputs, embeds)
        else:
            outputs = self.ad_norm1(inputs)

        outputs = self.sp_norm_conv1(self.relu1(outputs))
        if self.batchnorm:
            outputs = self.ad_norm2(outputs, embeds)
        else:
            outputs = self.ad_norm2(outputs)

        outputs = self.sp_norm_conv2(self.relu2(outputs))

        # skip connection
        outputs += self.skip_connection(inputs)

        if self.downsample:
            outputs = self.down(outputs)

        return outputs

class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO
        self.max_channels = max_channels
        self.use_class_condition = use_class_condition

        self.embed = torch.nn.Embedding(num_classes, noise_channels)

        if self.use_class_condition:
            noise_channels = noise_channels*2

        self.sp_norm_lin = torch.nn.utils.spectral_norm(nn.Linear(noise_channels, 4*4*self.max_channels))

        self.parb1 = PreActResBlock(self.max_channels, self.max_channels // 2, embed_channels=noise_channels, batchnorm=self.use_class_condition, upsample=True)
        self.parb2 = PreActResBlock(self.max_channels // 2, self.max_channels // 4, embed_channels=noise_channels, batchnorm=self.use_class_condition, upsample=True)
        self.parb3 = PreActResBlock(self.max_channels // 4, self.max_channels // 8, embed_channels=noise_channels, batchnorm=self.use_class_condition, upsample=True)
        self.parb4 = PreActResBlock(self.max_channels // 8, self.max_channels // 16, embed_channels=noise_channels, batchnorm=self.use_class_condition, upsample=True)
        
        self.head = nn.Sequential(
            nn.BatchNorm2d(min_channels),
            nn.ReLU(),
            torch.nn.utils.spectral_norm(nn.Conv2d(min_channels, 3, 3, padding=1)),
            nn.Sigmoid()
        )

    def forward(self, noise, labels):
        # TODO
        if self.use_class_condition: 
            noise = torch.cat((self.embed(labels), noise), dim=-1)  

        outputs = self.sp_norm_lin(noise).view(-1, self.max_channels, 4, 4)
        outputs = self.parb1(outputs, noise)
        outputs = self.parb2(outputs, noise)
        outputs = self.parb3(outputs, noise)
        outputs = self.parb4(outputs, noise)
            
        outputs = self.head(outputs)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs

class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO
        self.use_projection_head = use_projection_head

        self.head = nn.Sequential(
            spectral_norm(nn.Conv2d(3, min_channels, 3, padding=1)), 
            nn.ReLU(),
            nn.BatchNorm2d(min_channels)
        )
        
        self.parb1 = PreActResBlock(min_channels, min_channels * 2, downsample=True)
        self.parb2 = PreActResBlock(min_channels * 2, min_channels * 4, downsample=True)
        self.parb3 = PreActResBlock(min_channels * 4, min_channels * 8, downsample=True)
        self.parb4 = PreActResBlock(min_channels * 8, min_channels * 16, downsample=True)
        
        self.sp_norm_embed = torch.nn.utils.spectral_norm(nn.Embedding(max_channels, 1))
        self.sp_norm_lin = torch.nn.utils.spectral_norm(nn.Linear(max_channels, 1))
        
        self.relu = nn.ReLU()

    def forward(self, inputs, labels):
        # TODO
        outputs = self.head(inputs)
        
        outputs = self.parb1(outputs)
        outputs = self.parb2(outputs)
        outputs = self.parb3(outputs)
        outputs = torch.sum(self.relu(self.parb4(outputs)), dim=(2, 3))

        scores = self.sp_norm_lin(outputs) # TODO
        
        if self.use_projection_head:
            scores = scores + torch.sum(torch.mul(self.sp_norm_embed(labels), scores), dim=1, keepdim=True)
        
        scores = scores.view((inputs.shape[0],))

        assert scores.shape == (inputs.shape[0],)
        return scores