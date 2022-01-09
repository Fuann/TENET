#!/usr/bin/env python
# wujian@2018
# fuann@2020

import os
import argparse
import random
import torch

from libs.trainer import SnrTrainer, SiSnrTrainer, L1Trainer, PFPTrainer, HybridTrainer
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger

from conv_tas_net import ConvTasNet         # convtasnet (skip)
from conf import skip_conf
from dccrn import DCCRN                     # dccrn
from conf import dccrn_conf
from dcunet import DeepConvolutionalUNet    # dcunet
from conf import dcunet_conf
from dual_path_transformer import DPTNet    # dptnet
from conf import dpt_conf
from tenet import TENET                     # tenet
from conf import tenet_conf

from conf import trainer_conf, train_data, dev_data, chunk_size, fs

logger = get_logger(__name__)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))
    print("Use model: {}".format(args.model))

    if args.model == 'TCN-skip':
        nn_conf = skip_conf
        nnet = ConvTasNet(**nn_conf)
    elif args.model == 'DCCRN':
        nn_conf = dccrn_conf
        nnet = DCCRN(**nn_conf)
    elif args.model == 'PFPL':
        nn_conf = dcunet_conf
        nnet = DeepConvolutionalUNet(**nn_conf)
    elif args.model == 'DPTNet':
        nn_conf = dpt_conf
        nnet = DPTNet(**nn_conf)
    elif args.model == 'TENET':
        nn_conf = tenet_conf
        nnet = TENET(**nn_conf)

    if args.model == 'TENET':
        trainer = HybridTrainer(nnet, gpuid=gpuids, njobs=args.num_workers, checkpoint=args.checkpoint, resume=args.resume, **trainer_conf)
    elif args.model == 'PFPL':
        trainer = PFPTrainer(nnet, gpuid=gpuids, njobs=args.num_workers, checkpoint=args.checkpoint, resume=args.resume, **trainer_conf)
    else:
        trainer = SiSnrTrainer(nnet, gpuid=gpuids, njobs=args.num_workers, checkpoint=args.checkpoint, resume=args.resume, **trainer_conf)
        #trainer = SnrTrainer(nnet, gpuid=gpuids, njobs=args.num_workers, checkpoint=args.checkpoint, resume=args.resume, **trainer_conf)
        #trainer = L1Trainer(nnet, gpuid=gpuids, njobs=args.num_workers, checkpoint=args.checkpoint, resume=args.resume, **trainer_conf)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "chunk_size": chunk_size
    }
    for conf, fname in zip([nn_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, args.checkpoint, fname)

    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start TasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model",
                        type=str,
                        default="TCN-skip",
                        help="TCN-skip, DCCRN, DPTNet, TENET")
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to dump models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--batch-size",
                        type=int,
                        default=2,  #16
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers",
                        type=int,
                        default=1,
                        help="Number of workers used in data loader")
    args = parser.parse_args()
    #logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
