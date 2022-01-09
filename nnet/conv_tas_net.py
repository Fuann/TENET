# wujian@2018
# fuann@2020

import torch as th
import math
import torch.nn as nn
import torch.nn.functional as F
from libs.audio import WaveReader
from libs.stft import STFT, ISTFT

class PositionalEncoding(nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout, max_len=5000, reverse=False, mode="add"):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(th.tensor(0.0).expand(1, max_len))
        self.mode = mode

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = th.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = th.arange(
                x.size(1) - 1, -1, -1.0, dtype=th.float32
            ).unsqueeze(1)
        else:
            position = th.arange(0, x.size(1), dtype=th.float32).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, self.d_model, 2, dtype=th.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: th.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        # B x N x T -> B x T x N
        x = x.transpose(1, 2)
        self.extend_pe(x)
        pe = self.pe[:, : x.size(1)].expand_as(x)
        #print(x.shape, pe.shape)
        #input()
        if self.mode == "add":
            x = x * self.xscale + pe
        elif self.mode == "cat":
            x = th.cat((x, pe), 2)
        x = x.transpose(1, 2)
        return self.dropout(x)

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)

class Stft(STFT):
    """
    1D stft in ConvTasNet
    """
    def __init__(self, *args, **kwargs):
        super(Stft, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("accept 2/3D tensor as input")
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class iStft(ISTFT):
    """
    1D istft in ConvTasNet
    """
    def __init__(self, *args, **kwargs):
        super(iStft, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
            # NOTE: bug when data pararell with batch size 2
            x = th.unsqueeze(x, 0) if x.dim() == 1 else x
        return x

class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("accept 2/3D tensor as input")
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x

class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
            # NOTE: bug when data pararell with batch size 2
            x = th.unsqueeze(x, 0) if x.dim() == 1 else x
        return x

class BPF_encoder(nn.Module):
    def __init__(self, ipt_dim=1, stft_dim=256, conv_dim=256, hid_dim=128, 
                    kernel_size=16, stride=8, concat=False):
        super(BPF_encoder, self).__init__()
        self.stft = STFT(stft_dim, kernel_size, stride)
        self.conv1d = Conv1D(1, conv_dim, kernel_size, stride, bias=False)
        self.proj1 = Conv1D(stft_dim, hid_dim, 1, bias=True)
        self.proj2 = Conv1D(conv_dim, hid_dim, 1, bias=True)
        self.mask = Conv1D(hid_dim*2, hid_dim, 1, bias=True)
        self.concat = concat
        
    def forward(self, x):
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
        # n x N x T
        stft = self.stft(x)
        conv1d = self.conv1d(x)
        # fusion
        x1 = self.proj1(stft)
        x2 = self.proj2(conv1d)
        mask = F.sigmoid(self.mask(th.cat((x1, x2), 1)))
        fusion = mask * x1 + (1 - mask) * x2
        if self.concat:
            return th.cat((conv1d, stft, fusion), 1)
        return fusion


class GatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, padding=0, dilation=1):
        super(GatedConv1D, self).__init__()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                groups=groups, padding=padding, dilation=dilation, bias=True),
                                    GlobalChannelLayerNorm(out_channels, elementwise_affine=True),
                                    nn.PReLU())
        self.output_gate = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size,
                groups=groups, padding=padding, dilation=dilation, bias=True),
                                    GlobalChannelLayerNorm(out_channels, elementwise_affine=True),
                                    nn.Sigmoid())
    def forward(self, x):
        return self.output(x)*self.output_gate(x)

class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """
    def __init__(self,
                 in_channels=128,
                 conv_channels=512,
                 skip_channels=128,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # gated depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1)
        # skip conv
        if skip_channels:
            self.skip_conv = nn.Conv1d(conv_channels, in_channels, 1)
        self.skip_channels = skip_channels

        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        out = self.lnorm2(self.prelu2(y))

        res = self.sconv(out)
        if not self.skip_channels:
            return res

        skip = self.skip_conv(out)
        return res, skip

class TCN(nn.Module):
    """
    Input:  n x N x T
    Output: spk x [n x N x T] mask
    """
    def __init__(self, 
                 N=512,
                 X=8,
                 R=3,
                 B=128,
                 H=512,
                 S=128,
                 P=3,
                 norm="cLN",
                 num_spks=2,
                 non_linear="relu",
                 causal=False):
        super(TCN, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax,
            "none": None
        }
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        self.skip_channels = S
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        self.blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                self.blocks.append(Conv1DBlock(
                    in_channels=B,
                    conv_channels=H,
                    skip_channels=S,
                    kernel_size=P,
                    norm=norm,
                    causal=causal,
                    dilation=2**x))
        mask_ipt = S if S else B
        # n x B x T => n x 2N x T
        self.mask = nn.Sequential(
            nn.PReLU(),
            #GatedConv1D(mask_ipt, mask_ipt, 1),
            Conv1D(mask_ipt, num_spks * N, 1))
        self.num_spks = num_spks

    def forward(self, x):
        # n x B x T
        output = self.proj(self.ln(x))

        # repeats blocks
        skip_connection = 0.
        for i, block in enumerate(self.blocks):
            tcn_out = block(output)
            if self.skip_channels:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual

        # Use residual output when no skip connection
        mask_ipt = skip_connection if self.skip_channels else output

        # n x 2N x T
        e = th.chunk(self.mask(mask_ipt), self.num_spks, 1)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        elif self.non_linear_type == "relu" or self.non_linear_type == "sigmoid":
            m = self.non_linear(th.stack(e, dim=0))
        else:
            m = th.stack(e, dim=0)
        
        # spks x [n x N x T]
        s = [x * m[n] for n in range(self.num_spks)]
        return s

class ConvTasNet(nn.Module):
    def __init__(self,
                 L=16,      #20
                 N=512,     #256
                 X=8,
                 R=3,       #3
                 B=128,     #256
                 H=512,
                 S=128,
                 P=3,
                 norm="gLN",
                 num_spks=2,
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax,
            "none": None
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, kernel_size=L, stride=L//2)
        self.decoder_1d = ConvTrans1D(N, 1, kernel_size=L, stride=L//2)
        self.separate = TCN(N, X, R, B, H, S, P, norm, num_spks, non_linear, causal)
        self.num_spks = num_spks

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        # NOTE: Encoder 
        # n x 1 x S => n x N x T
        w = self.encoder_1d(x)
        w = F.relu(w)

        # n x B x T
        # NOTE: Separate
        s =  self.separate(w)

        # NOTE: Decoder
        out = self.decoder_1d(s[0], squeeze=True)
        # spks x n x S
        return out, s[0]

def foo_conv1d_block():
    x = th.rand(4, 128, 6000)
    nnet = Conv1DBlock()
    print(param(nnet))
    res, skip = nnet(x)
    print(res.shape, skip.shape)

def foo_conv_tas_net():
    nnet = ConvTasNet(norm="gLN", R=3, causal=False)
    print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    x = th.rand(4, 80000)
    print("Raw: ", x.shape)
    x = nnet(x)
    s1, s2 = x[0], x[1]
    print("Sep: ", s1.shape)

if __name__ == "__main__":
    #x = th.rand(4, 256, 99)
    #y = th.rand(4, 512, 99)
    #x = th.rand(4, 1000)
    #k = nnet(x)
    #print(k.shape)

    foo_conv_tas_net()
    #foo_conv1d_block()
    #foo_layernorm()
