import torch
import torch.nn as nn

class ConvTrack(nn.Module):
    def __init__(self, channels: int, compression_ratio: int, activation: str = 'Tanh'):
        super(ConvTrack, self).__init__()
        
        mid_channels = int(channels/compression_ratio)
        conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=mid_channels,
            kernel_size = 3,
            padding='same')
        conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=channels,
            kernel_size=3,
            padding='same')

        conv2_act = getattr(nn, activation)()

        self.conv_track = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            conv2_act
            )

    def forward(self, x):
        return(self.conv_track(x))


class DimensionalAttentionBlock(nn.Module):
    def __init__(self, channels: int, compression_ratio: int, activation: str = 'Tanh'):
        super(DimensionalAttentionBlock, self).__init__()

        # need three tracks, one for each dimension
        self.d_track = ConvTrack(channels, compression_ratio, activation)
        self.h_track = ConvTrack(channels, compression_ratio, activation)
        self.w_track = ConvTrack(channels, compression_ratio, activation)

        # need channels for reconstruction code
        self.channels = channels
        

    def reduction(self, x):
        ''' Deconstruct 3D input to 2D outputs for each dimension via max pooling.
        x: NxCxDxHxW tensor

        returns: 
            d: NxCxHxW tensor
            h: NxCxDxW tensor
            w: NxCxDxH tensor
        '''
        d, _ = torch.max(x, dim=2)
        h, _ = torch.max(x, dim=3)
        w, _ = torch.max(x, dim=4)
        return d, h, w

    def reconstruction(self, d, h, w):
        ''' Reconstruct 3D output from 2D inputs. 3D vox value == mean of each value at that location
        d: NxCxWxH tensor
        h: NxCxDxW tensor
        w: NxCxDxH tensor

        returns: NxCxDxHxW
        '''
        # create 3D tensor that repopulates feature map f along the original f dimension
        # add single dimension, then expand over that dimension; 2 is d, 3 is h, 4 is w
        dims = [d.shape[0], d.shape[1], w.shape[2], d.shape[2], d.shape[3]] # NxCxDxHxW
        a = d.unsqueeze(2).expand(dims)
        b = h.unsqueeze(3).expand(dims)
        c = w.unsqueeze(4).expand(dims)
        return (a+b+c)/3

    def forward(self, x):
        # 3D -> 3 2D
        d, h, w = self.reduction(x)
        # Attention learning convolutions
        d = self.d_track(d)
        h = self.h_track(h)
        w = self.w_track(w)
        # 2D -> 3D
        attentions = self.reconstruction(d, h, w)
        # Apply attention scores to input
        out = (attentions + 1) * x
        return out
