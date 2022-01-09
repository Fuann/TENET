#!/usr/bin/env python3

# wujian@2018

import argparse
import os

import numpy as np

from libs.utils import stft, istft, get_logger
from libs.opts import StftParser
from libs.data_handler import SpectrogramReader, NumpyReader, ScriptReader, WaveWriter, ArchiveWriter

logger = get_logger(__name__)


def run(args):

    feat_reader = ScriptReader(args.feats_scp)
    MaskReader = {"numpy": NumpyReader, "kaldi": ScriptReader}
    mask_reader = MaskReader[args.fmt](args.mask_scp)

    JOB = args.feats_scp.split('.')[1]

    saveArk = args.enhan + "/" + "enhanced." + JOB + ".ark"
    saveScp = args.enhan + "/" + "enhanced." + JOB + ".scp"

    num_done = 0
    with ArchiveWriter(ark_path=saveArk, scp_path=saveScp) as writer:
        for key, feats in feat_reader:
            if key in mask_reader:
                num_done += 1
                mask = mask_reader[key]
                # mask sure mask in T x F
                _, F = feats.shape
                if mask.shape[0] == F:
                    mask = np.transpose(mask)
                logger.info("Processing utterance %s...", key)
                if mask.shape != feats.shape:
                    raise ValueError(
                        "Dimention mismatch between mask and feats" +
                        "(%d x %d vs " +
                        "%d x %d), need " +
                        "check configures", mask.shape[0], mask.shape[1], feats.shape[0], feats.shape[1])

                enh_feats = feats * mask
                writer.write(key, enh_feats)
    logger.info(
        "Processed %d utterances over %d", num_done, len(feat_reader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to separate target component at feature domain from mixtures given Tf-masks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument("feats_scp",
                        type=str,
                        help="Mixture feats in kaldi format")
    parser.add_argument("mask_scp",
                        type=str,
                        help="Scripts of masks in kaldi's "
                        "archive or numpy's ndarray")
    parser.add_argument("enhan",
                        type=str,
                        help="Location to dump enhanced feats ark,scp")
    parser.add_argument("--mask-format",
                        dest="fmt",
                        choices=["kaldi", "numpy"],
                        default="kaldi",
                        help="Define format of masks, kaldi's "
                        "archives or numpy's ndarray")


    args = parser.parse_args()
    run(args)
