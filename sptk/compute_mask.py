#!/usr/bin/env python

# wujian@2018
"""
Compute IAM(FFT-mask,SMM)/IBM/IRM/PSM masks, using as training targets
"""

import argparse
import numpy as np

from libs.data_handler import SpectrogramReader, ArchiveWriter
from libs.utils import get_logger, cmat_abs, EPSILON
from libs.opts import StftParser
from libs.exraw import BinaryWriter

logger = get_logger(__name__)


def sigmoid(x, a=1, b=0):
    """
    Use sigmoid to compress complex mask, avoid overflow
    To uncompress:
    def sigmoid_inv(m, a=1, b=0):
        m = np.maximum(m, EPSILON)
        x = np.maximum(1 / m - 1, EPSILON)
        return (b - np.log(x)) / a
    """
    s = np.zeros_like(x)
    m = (x >= 0)
    # 1) x >= 0
    e = np.exp(-x[m] * a + b)
    s[m] = 1 / (1 + e)
    # 2) x < 0
    e = np.exp(x[~m] * a + b)
    s[~m] = e / (1 + e)
    return s


def tangent(x, K=10, C=0.1):
    """
    Use tangent to compress complex mask, avoid overflow
    To uncompress:
    def tangent_inv(m, K=10, C=0.1):
        x = (K - m) / np.maximum(EPSILON, K + m)
        return -np.log(np.maximum(x, EPSILON)) / C
    """
    s = np.zeros_like(x)
    m = (x >= 0)
    # 1) x >= 0
    e = np.exp(-x[m] * C)
    s[m] = K * (1 - e) / (1 + e)
    # 2) x < 0
    e = np.exp(x[~m] * C)
    s[~m] = K * (e - 1) / (e + 1)
    return s


def compute_mask(speech, noise_or_mixture, mask):
    """
    for signal model:
        y = x1 + x2
    def f = STFT(x):
        f(y) = f(x1) + f(x2) => |f(y)| = |f(x1) + f(x2)| < |f(x1)| + |f(x2)|
    for irm:
        1) M(x1) = |f(x1)| / (|f(x1)| + |f(x2)|)            DongYu
        2) M(x1) = |f(x1)| / sqrt(|f(x1)|^2 + |f(x2)|^2)    DeliangWang
        s.t. 1 >= 2) >= 1) >= 0
    for iam(FFT-mask, smm):
        M(x1) = |f(x1)| / |f(y)| = |f(x1)| / |f(x1) + f(x2)| in [0, oo]
    for psm:
        M(x1) = |f(x1) / f(y)| = |f(x1)| * cos(delta_phase) / |f(y)|
    for crm:
        M(x1) = f(x1) / f(y)
    """
    if mask == "ibm":
        binary_mask = cmat_abs(speech) > cmat_abs(noise_or_mixture)
        return binary_mask.astype(np.float)
    # irm/iam/psm
    if mask == "irm":
        # denominator = cmat_abs(speech) + cmat_abs(noise_or_mixture)
        denominator = np.sqrt(
            cmat_abs(speech)**2 + cmat_abs(noise_or_mixture)**2)
    elif mask == "crm":
        denominator = noise_or_mixture + EPSILON
    else:
        denominator = cmat_abs(noise_or_mixture)
    if mask == "psm":
        return cmat_abs(speech) * np.cos(
            np.angle(noise_or_mixture) - np.angle(speech)) / denominator
    elif mask == "psa":
        # keep nominator only
        return cmat_abs(speech) * np.cos(
            np.angle(noise_or_mixture) - np.angle(speech))
    elif mask == "crm":
        # stack real/imag part
        cpx_mask = speech / denominator
        return np.hstack(
            [tangent(np.real(cpx_mask)),
             tangent(np.imag(cpx_mask))])
    else:
        # irm/iam/smm
        return cmat_abs(speech) / denominator


def run(args):
    # shape: T x F, complex
    stft_kwargs = {
        "frame_len": args.frame_len,
        "frame_hop": args.frame_hop,
        "round_power_of_two": args.round_power_of_two,
        "window": args.window,
        "center": args.center,  # false to comparable with kaldi
    }

    speech_reader = SpectrogramReader(args.speech_scp, **stft_kwargs)
    denorm_reader = SpectrogramReader(args.denorm_scp, **stft_kwargs)

    num_utts = 0
    cutoff = args.cutoff
    WriterImpl = {"kaldi": ArchiveWriter, "exraw": BinaryWriter}[args.format]

    with WriterImpl(args.mask_ark, args.scp) as writer:
        for key, speech in speech_reader:
            if key in denorm_reader:
                num_utts += 1
                denorm = denorm_reader[key]
                mask = compute_mask(speech[0] if speech.ndim == 3 else speech,
                                    denorm[0] if denorm.ndim == 3 else denorm,
                                    args.mask)
                # iam, psm, psa
                if cutoff > 0:
                    num_items = np.sum(mask > cutoff)
                    mask = np.minimum(mask, cutoff)
                    if num_items:
                        percent = float(num_items) / mask.size
                        logger.info(
                            "Clip {:d}({:.2f}) items over {:.2f} for utterance {}"
                            .format(num_items, percent, cutoff, key))
                num_items = np.sum(mask < 0)
                # psm, psa
                if num_items:
                    percent = float(num_items) / mask.size
                    average = np.sum(mask[mask < 0]) / num_items
                    logger.info(
                        "Clip {:d}({:.2f}, {:.2f}) items below zero for utterance {}"
                        .format(num_items, percent, average, key))
                    mask = np.maximum(mask, 0)
                writer.write(key, mask)
            else:
                logger.warn("Missing bg-noise for utterance {}".format(key))
    logger.info("Processed {} utterances".format(num_utts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute Tf-mask(as targets for Kaldi's nnet3, "
        "only for 2 component case, egs: speech & noise)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[StftParser.parser])
    parser.add_argument("speech_scp",
                        type=str,
                        help="Scripts for target speech in Kaldi format")
    parser.add_argument("denorm_scp",
                        type=str,
                        help="Scripts for bg-noise or mixture in Kaldi format")
    parser.add_argument("mask_ark",
                        type=str,
                        help="Location to dump mask archives")
    parser.add_argument("--format",
                        type=str,
                        default="kaldi",
                        choices=["kaldi", "exraw"],
                        help="Output archive format, see "
                        "format in sptk/libs/exraw.py")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate "
                        "corresponding mask scripts")
    parser.add_argument("--mask",
                        type=str,
                        default="irm",
                        choices=["irm", "ibm", "smm", "psm", "psa", "crm"],
                        help="Type of masks(irm/ibm/smm(FFT-mask,"
                        "smm)/psm/psa/crm) to compute. \'psa\' is not "
                        "a real mask(nominator of psm), but could compute "
                        "using this command. Noted that if "
                        "iam/psm assigned, second .scp is expected "
                        "to be noisy component.")
    parser.add_argument("--cutoff",
                        type=float,
                        default=-1,
                        help="Cutoff values(<=0, not cutoff) for "
                        "some non-bounded masks, "
                        "egs: iam/psm")
    args = parser.parse_args()
    run(args)
