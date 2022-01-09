# wujian@2018
# fuann@2020

import os
import sys
import time

from itertools import permutations
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchcontrib.optim import SWA
from torch.nn.utils import clip_grad_norm_
from pesq import pesq
from multiprocessing import Pool
from .utils import get_logger
from .pfp_loss import PerceptualLoss

MAX_INT16 = np.iinfo(np.int16).max

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)

def cal_sisnr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))

    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    t = th.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def cal_snr(x, s, eps=1e-8):
    """
    Arguments:
    x: separated signal, N x S tensor
    s: reference signal, N x S tensor
    Return:
    snr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    return 20 * th.log10(eps + l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps))

def cal_pesq(x, y):
    """
    x: deg
    y: ref
    """
    try:
        score = pesq(16000, y, x, 'wb')
    except:
        score = 0.
    return score

def evaluate(x, y, fn, n_jobs=4):
    y = list(y.cpu().detach().numpy())
    x = list(x.cpu().detach().numpy())
    pool = Pool(processes=n_jobs)
    try:
        ret = pool.starmap(
            fn, 
            iter([(deg[:ref.size], ref) for deg, ref in zip(x, y)])
        )
        pool.close()
        return th.FloatTensor(ret).mean()

    except KeyboardInterrupt:
        pool.terminate()
        pool.close()    

class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.4f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class Trainer(object):
    def __init__(self,
                 nnet,
                 seed=1016,
                 checkpoint="checkpoint",
                 njobs=10,
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 optimizer_swa=False,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=6,
                 model_type=None,
                 loss_type=None,
                 pretrained_model_path=None,
                 alpha=None,
                 beta=None,
                 gamma=None):

        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exit")
        # set seed
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False

        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        #self.device = th.device("cuda:{}".format(gpuid[0]))
        self.device = th.device("cuda")
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.njobs = njobs
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        # perceptual loss related
        self.model_type = model_type
        self.loss_type = loss_type
        self.pretrained_model_path = pretrained_model_path
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # ddp to cuda
            nnet = nn.DataParallel(nnet)
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            # ddp to cuda
            nnet = nn.DataParallel(nnet)
            self.nnet = nnet.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        if optimizer_swa:
            self.optimizer = SWA(self.optimizer, 
                swa_start=10, swa_freq=5, swa_lr=0.05).optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adamw": th.optim.AdamW,
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            #### NOTE train loss ####
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                #### NOTE eval loss ####
                loss = self.compute_loss(egs, valid=True)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=50):
        # avoid alloc memory from gpu0
        #with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # check if save is OK
            self.save_checkpoint(best=False)
            cv = self.eval(dev_loader)
            best_loss = cv["loss"]
            self.logger.info("START FROM EPOCH {:d}, LOSS = {:.2f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0
            # make sure not inf
            self.scheduler.best = best_loss
            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                stats["tr"] = "train = {:+.4f} ({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                stats["cv"] = "dev = {:+.2f} ({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.2f}".format(
                        self.scheduler.best)
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                # schedule here
                self.scheduler.step(cv["loss"])
                # flush scheduler info
                sys.stdout.flush()
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))

class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, egs, valid=False):
        # n x S
        #ests = th.nn.parallel.data_parallel(
        #    self.nnet, egs["mix"], device_ids=self.gpuid)
        ests = self.nnet(egs['mix'])

        # [clean, noise] x n x S
        ref = egs["ref"]
        
        if valid:
            """ Valid: PESQ-WB loss
            """
            pesq_loss = -1 * evaluate(ests[0], ref, fn=cal_pesq, n_jobs=self.njobs)

            # pesq_loss
            return pesq_loss
        else:
            """ Train: SISNR loss
            """
            sisnr_loss = -th.mean(cal_sisnr(ests[0], ref))
            return sisnr_loss

class SnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SnrTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, egs, valid=False):
        # n x S
        ests = self.nnet(egs['mix'])
        # 2 x n x S, [0] for clean, [1] for noise
        ref = egs["ref"]

        if valid:
            """ Valid: PESQ-WB loss
            """
            pesq_loss = -1 * evaluate(ests[0], ref, fn=cal_pesq, n_jobs=self.njobs)
            return pesq_loss
        else:
            """ Train: SNR loss
            """
            snr_loss = -th.mean(cal_snr(ests[0], ref))
            return snr_loss

class L1Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(L1Trainer, self).__init__(*args, **kwargs)

    def compute_loss(self, egs, valid=False):
        # n x S
        ests = self.nnet(egs['mix'])

        # [clean, noise] x n x S
        ref = egs["ref"]

        if valid:
            """ Valid: PESQ-WB loss
            """
            pesq_loss = -1 * evaluate(ests[0], ref, fn=cal_pesq, n_jobs=self.njobs)
            return pesq_loss
        else:
            """ Train: L1 loss
            """
            l1_loss = F.l1_loss(ests[0], ref)
            return l1_loss

class PFPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(PFPTrainer, self).__init__(*args, **kwargs)
        # large-wav2vec
        pfp_loss = PerceptualLoss(
            model_type=self.model_type,
            loss_type=self.loss_type,
            PRETRAINED_MODEL_PATH=self.pretrained_model_path,
            device=self.device
        )
        # criterion - pfp + MAE
        self.criterion = lambda y_hat, y: pfp_loss(y_hat, y) + F.l1_loss(y_hat, y)

    def compute_loss(self, egs, valid=False):
        # n x S
        #ests = th.nn.parallel.data_parallel(
        #    self.nnet, egs["mix"], device_ids=self.gpuid)
        ests = self.nnet(egs['mix'])

        # [clean, noise] x n x S
        ref = egs["ref"]

        # PFP LOSS
        if valid:
            """ Valid: PESQ-WB loss
            """
            pesq_loss = -1 * evaluate(ests[0], ref, fn=cal_pesq, n_jobs=self.njobs)
            return pesq_loss
        else:
            pfp_loss = self.criterion(ests[0], ref)
            return pfp_loss

class HybridTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(HybridTrainer, self).__init__(*args, **kwargs)
        # large-wav2vec
        pfp_loss = PerceptualLoss(
            model_type=self.model_type,
            loss_type=self.loss_type,
            PRETRAINED_MODEL_PATH=self.pretrained_model_path,
            device=self.device
        )
        # criterion
        self.criterion = lambda y_hat, y: pfp_loss(y_hat, y)
        self.a = self.alpha
        self.b = self.beta
        self.r = self.gamma

    def compute_loss(self, egs, valid=False):
        # n x S
        #ests, z, p  = th.nn.parallel.data_parallel(
        #    self.nnet, (egs["mix"], egs["mix2"]), device_ids=self.gpuid)
        ests, z, p = self.nnet(egs['mix'], egs['mix2'])
        
        # [clean, noise] x n x S
        ref = egs["ref"]
        ref2 = egs["ref2"]

        if valid:
            """ Valid: PESQ-WB loss
            """
            pesq_loss = -1 * evaluate(ests[0], ref, fn=cal_pesq, n_jobs=self.njobs)
            return pesq_loss
        else:
            """ Train: sisnr + a * pfp loss
            """
            # NOTE: reconstruction loss
            # sisnr + pfp
            rc1 = -th.mean(cal_sisnr(ests[0], ref)) + self.a * self.criterion(ests[0], ref)
            rc2 = -th.mean(cal_sisnr(ests[1], ref2)) + self.a * self.criterion(ests[1], ref2)
            rc_loss = self.b * rc1 + self.r * rc2
            #print("rc_loss1: {:8.4f},  rc_loss2: {:8.4f}".format(rc_loss1.item(), rc_loss2.item()))
            return rc_loss
            
