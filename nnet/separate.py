#!/usr/bin/env python
# wujian@2018
# fuann@2020

import os
import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm

from conv_tas_net import ConvTasNet         # skip
from dccrn import DCCRN                     # dccrn
from dcunet import DeepConvolutionalUNet    # dcunet
from dual_path_transformer import DPTNet    # DPTNet
from tenet import TENET                     # TENET

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, write_wav

MAX_INT16 = np.iinfo(np.int16).max

logger = get_logger(__name__)

class NnetComputer(object):
    def __init__(self, cpt_dir, model, gpuid):
        self.device = th.device(
            "cuda:{}".format(gpuid)) if gpuid >= 0 else th.device("cpu")
        nnet = self._load_nnet(cpt_dir, model)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.model = model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir, model):
        if model == 'TCN-skip':
            nnet = ConvTasNet(**nnet_conf)
        elif model == 'DCCRN':
            nnet = DCCRN(**nnet_conf)
        elif model == 'PFPL':
            nnet = DeepConvolutionalUNet(**nnet_conf)
        elif model == 'DPTNet':
            nnet = DPTNet(**nnet_conf)
        elif model == 'TENET':
            nnet = TENET(**nnet_conf)

        # load state
        nnet = nn.DataParallel(nnet)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        
        return nnet

    def compute(self, samps):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            raw = raw.unsqueeze(0) if raw.dim() == 1 else raw
            if self.model == 'TENET':
                sps = self.nnet(raw, eval_mode=True)
            else:
                sps, feat = self.nnet(raw)
            sp_samps = np.squeeze(sps.cpu().data.numpy())

            #th.manual_seed(10)
            #th.autograd.set_detect_anomaly(True)
            # spec, wav
            #sps = self.nnet(raw.unsqueeze(1))[1]
            #sp_samps = [np.squeeze(sps.detach().cpu().numpy())]
            return sp_samps

def run(args):
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.model, args.gpu)

    start = time.time()
    for i, (key, mix_samps) in enumerate(tqdm(mix_input, ascii=True)):
        if os.path.exists(os.path.join(args.dump_dir, "enhance/{}.wav").format(key)):
            continue
        samps = computer.compute(mix_samps)
        norm = np.linalg.norm(mix_samps, np.inf)
        samps = samps[:mix_samps.size]
        # norm
        samps = samps * norm / np.max(np.abs(samps))
        if not os.path.exists(os.path.join(args.dump_dir, "enhance/{}.wav").format(key)):
            write_wav(
                os.path.join(args.dump_dir, "enhance/{}.wav".format(key)),
                samps,
                fs=args.fs
            )
        else:
            continue
    print( "Execution Time:", time.time() - start,  "(sec)" )
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--model", type=str, default='TCN-skip', required=True, help="[TCN-skip/DPTNet/TENET]")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=16000, help="Sample rate for mixture input")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
