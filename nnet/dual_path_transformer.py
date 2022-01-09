from math import ceil
import warnings

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
from libs.MultiHeadAttention import MultiHeadAttention
from libs.overlap_add import DualPathProcessing
from libs import activations, norms
from libs.utils import check_parameters
from conv_tas_net import Conv1D, ConvTrans1D, Stft, iStft, BPF_encoder

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
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x

class ImprovedTransformedLayer(nn.Module):
    """
        Improved Transformer module as used in [1].
        It is Multi-Head self-attention followed by LSTM, activation and linear projection layer.
        Args:
            embed_dim (int): Number of input channels.
            n_heads (int): Number of attention heads.
            dim_ff (int): Number of neurons in the RNNs cell state.
                Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
            dropout (float, optional): Dropout ratio, must be in [0,1].
            activation (str, optional): activation function applied at the output of RNN.
            bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
                (Intra-Chunk is always bidirectional).
            norm (str, optional): Type of normalization to use.
            
        nn.ModuleList(
            [   
                ImprovedTransformedLayer(
                    self.mha_in_dim, self.n_heads, self.ff_hid, self.dropout, self.ff_activation, True, self.norm_type, self.sparse_topk,                            
                ),
                ImprovedTransformedLayer(
                    self.mha_in_dim, self.n_heads, self.ff_hid, self.dropout, self.ff_activation, True, self.norm_type, self.sparse_topk,                            
                ),
            ]
        )
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        dim_ff,
        dropout=0.0,
        mha_activation="relu",
        activation="relu",
        bidirectional=True,
        norm="gLN",
        sparse_topk=None,
    ):
        super(ImprovedTransformedLayer, self).__init__()

        # online-mha
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm_mha = norms.get(norm)(embed_dim)

        self.recurrent = nn.LSTM(embed_dim, dim_ff, 
            bidirectional=bidirectional, batch_first=True, num_layers=1)
        #self.recurrent = nn.GRU(embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True)
        ff_inner_dim = 2 * dim_ff if bidirectional else dim_ff

        self.activation = activations.get(activation)()
        self.linear = nn.Linear(ff_inner_dim, embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)

    def forward(self, x):
        # x is batch, channels, seq_len
        
        # online-mha: batch, seq_len, channels
        tomha = x.permute(0, 2, 1)

        # self-attention is applied
        out, attn = self.mha(tomha, tomha, tomha)
        # online-mha
        x = self.dropout(out.permute(0, 2, 1)) + x

        x = self.norm_mha(x)

        # rnn is applied
        #self.recurrent.flatten_parameters()
        out = self.linear(self.dropout(self.activation(self.recurrent(x.transpose(1, -1))[0])))
        x = self.dropout(out.transpose(1, -1)) + x

        return self.norm_ff(x)

class DPTransformer(nn.Module):
    """Dual-path Transformer introduced in [1].
    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        n_heads (int): Number of attention heads.
        ff_hid (int): Number of neurons in the RNNs cell state.
            Defaults to 256.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        ff_activation (str, optional): activation function applied at the output of RNN.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].
    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        in_chan=64,
        dpt_chan=64,
        n_src=2,
        n_heads=4,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        n_intra=1,
        n_inter=1,
        norm_type="gLN",
        mha_activation="relu",
        ff_activation="relu",
        mask_act="relu",
        mask_floor=None,
        bidirectional=True,
        dropout=0,
        sparse_topk=None,
    ):
        super(DPTransformer, self).__init__()
        self.in_chan = in_chan
        self.dpt_chan = dpt_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_intra = n_intra
        self.n_inter = n_inter
        self.n_src = n_src
        self.norm_type = norm_type
        self.mha_activation = mha_activation
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.mask_floor = mask_floor
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sparse_topk = sparse_topk

        self.in_norm = norms.get(norm_type)(self.in_chan)
        self.in_act = nn.PReLU()
        self.ola = DualPathProcessing(self.chunk_size, self.hop_size)
        # NOTE: input chunk projection
        self.first_in = nn.Sequential(
            #norms.get(norm_type)(self.in_chan),
            nn.Conv2d(self.in_chan, self.dpt_chan, kernel_size=1),
            nn.PReLU()
        )
        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [   
                        nn.Sequential(*[
                            *[
                                ImprovedTransformedLayer(
                                    self.dpt_chan, self.n_heads, self.ff_hid, self.dropout, self.mha_activation, self.ff_activation, self.bidirectional, self.norm_type, self.sparse_topk,                            
                                ) for _ in range(self.n_intra)
                            ]
                        ]),
                        nn.Sequential(*[
                            *[
                                ImprovedTransformedLayer(
                                    self.dpt_chan, self.n_heads, self.ff_hid, self.dropout, self.mha_activation, self.ff_activation, self.bidirectional, self.norm_type, self.sparse_topk,                            
                                ) for _ in range(self.n_inter)
                            ]
                        ]),
                    ]
                )
            )
        self.first_out = nn.Sequential(
            nn.PReLU(), 
            nn.Conv2d(self.dpt_chan, n_src * self.in_chan, 1)
        )
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())

        # Get activation function.
        if mask_act == "learned_sigmoid":
            #from libs.activations import LearnedSigmoid
            #self.output_act = LearnedSigmoid(self.in_chan)
            from libs.activations import Learnable_sigmoid
            self.output_act = Learnable_sigmoid(self.in_chan)
        else:
            if mask_act:
                mask_nl_class = activations.get(mask_act)
                self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        r"""Forward.
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$
        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]
        mixture_w = self.in_act(mixture_w)
        n_orig_frames = mixture_w.shape[-1]

        mixture_w = self.ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        mixture_w = self.first_in(mixture_w)
        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = self.ola.intra_process(mixture_w, intra)
            mixture_w = self.ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        output = self.ola.fold(output, output_size=n_orig_frames)

        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)

        est_mask = self.output_act(output).clamp(min=self.mask_floor) if self.mask_act else output
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "dpt_chan": self.dpt_chan,
            "ff_hid": self.ff_hid,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_intra": self.n_intra,
            "n_inter": self.n_inter,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "mask_floor": self.mask_floor,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }
        return config

class DPTNet(nn.Module):
    def __init__(self,
                feat_type='freq',
                n_filters=64,
                kernel_size=16,
                kernel_shift=8,
                n_feats=64,
                n_src=2,
                n_heads=8,
                ff_hid=128,
                mha_activation="relu",
                ff_activation="relu",
                chunk_size=100,
                hop_size=None,
                n_repeats=6,
                n_intra=1,
                n_inter=1,
                norm_type="gLN",
                mask_act='sigmoid',
                mask_floor=0.0,
                bidirectional=True,
                dropout=0,
                sparse_topk=None):
        super(DPTNet, self).__init__()

        # time domain
        if feat_type == 'time':
            self.encoder = Conv1D(1, n_filters, kernel_size=kernel_size, stride=kernel_shift)
            self.decoder = ConvTrans1D(n_filters, 1, kernel_size=kernel_size, stride=kernel_shift)
        # freq domain
        elif feat_type == 'freq':
            self.encoder = Stft(n_filters, kernel_size, kernel_shift, win_type="hanning")
            self.decoder = iStft(n_filters, kernel_size, kernel_shift, win_type="hanning")
        elif feat_type == 'merge':
        # cross domain
            self.encoder = BPF_encoder(
                    stft_dim=n_filters, conv_dim=n_filters, hid_dim=128, 
                    kernel_size=kernel_size, stride=kernel_shift, concat=True,
            )
            self.decoder = ConvTrans1D(n_filters*2+128, 1, kernel_size=kernel_size, stride=kernel_shift)
        else:
            raise RuntimeError("Unsupported feat_type: {}", format(feat_type))
            
        feat_dim = n_filters*2+128 if feat_type == 'merge' else n_filters

        # mask_net
        self.mask_net = DPTransformer(
            in_chan=feat_dim,
            dpt_chan=n_feats,
            n_src=n_src,
            n_heads=n_heads,
            ff_hid=ff_hid,
            chunk_size=chunk_size,
            hop_size=None,
            n_repeats=n_repeats,
            n_intra=n_intra,
            n_inter=n_inter,
            norm_type=norm_type,
            mha_activation=mha_activation,
            ff_activation=ff_activation,
            mask_act=mask_act,
            mask_floor=mask_floor,
            bidirectional=bidirectional,
            dropout=dropout,
            sparse_topk=sparse_topk,
        )
        self.n_src = n_src

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)

        # NOTE: encode
        x = self.encoder(x)
        # NOTE: Mask Net
        m = torch.chunk(self.mask_net(x), self.n_src, 1)
        # spks x [n x N x T]
        s = [x * m[n].squeeze(1) for n in range(self.n_src)]
        enh_feat = s[0]
        # NOTE: Decoder output 1
        est = self.decoder(enh_feat, squeeze=True)
        return est, enh_feat

if __name__ == "__main__":
    nnet = DPTNet()
    x = torch.rand(2, 64000)
    x = nnet(x)
    print(x[0].shape, x[1].shape, check_parameters(nnet))
