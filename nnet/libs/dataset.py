# wujian@2018
# fuann@2020

import random
import torch as th
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from pysndfx import AudioEffectsChain

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from .audio import WaveReader, write_wav, synthesis_noisy_y, flip_samples, mask_samples, swap_samples
import sys

def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    dataset = Dataset(**data_kwargs)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers)

class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, noise_scp=None, noise_snrs=[0, 5, 10, 15], sample_rate=16000, 
                    speed_perturb=False, flip=False, mask=False, shift=False, reverb=False):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate) # noisy
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate) # clean
        # sythesis another noisy wav
        #self.noise = WaveReader(noise_scp, sample_rate=sample_rate) if noise_scp else None # extra_noise
        #self.snrlist = noise_snrs

        # Audio effect
        self.speed_perturb = speed_perturb
        self.speed = (0.95, 1.05)   #0.95, 1.05 # 0.90, 1.10
        # flipping
        self.flip = flip
        self.frame_wise_flip = False
        self.flip_length = 500  # if frame-wise flip
        # masking
        self.mask = mask
        self.mask_len = 10   #10
        self.max_mask = 150  #160
        # shift
        self.shift = shift
        self.shift_len = 10000   #8000
        #self.shift_set_zero = False
        # reverb
        self.reverb = reverb

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]

        # speed perturb audio effect
        if self.speed_perturb:
            c_speed = random.uniform(*self.speed)
        else:
            c_speed = 1.0
        mix = self.random_data_augmentation(mix, c_speed)
        ref = self.random_data_augmentation(ref, c_speed)
            
        # flip samples
        if self.flip:
            mix2 = flip_samples(mix, flip_length=self.flip_length, frame_wise_flip=self.frame_wise_flip)
            ref2 = flip_samples(ref, flip_length=self.flip_length, frame_wise_flip=self.frame_wise_flip)
        else:
            mix2 = mix
            ref2 = ref
        # mask samples
        if self.mask:
            mix = mask_samples(mix, mask_length=self.mask_len, max_mask=self.max_mask, 
                replace_with_zero=True)
            mix2 = mask_samples(mix2, mask_length=self.mask_len, max_mask=self.max_mask, 
                replace_with_zero=True)
        # shift samples
        if self.shift:
            s1, s2 = np.random.randint(0, self.shift_len, size=2)
            mix, ref = np.roll(mix, s1), np.roll(ref, s1)
            mix2, ref2 = np.roll(mix2, s2), np.roll(ref2, s2)
            #if self.shift_set_zero:
            #    for raw in mix, ref:
            #        raw[:s1] = 0
            #    for rev in mix2, ref2:
            #        rev[:s2] = 0

        return {
            "mix": mix.astype(np.float32),
            "mix2": mix2.astype(np.float32),
            "ref": ref.astype(np.float32),
            "ref2": ref2.astype(np.float32)
        }

    def random_data_augmentation(self, signal, c_speed):
        # audio effect
        AE = AudioEffectsChain()
        if self.speed_perturb:
            AE = AE.speed(c_speed)
        if self.reverb:
            AE = AE.reverb()
        fx = (AE)
        signal = fx(signal)
        return signal


class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,         noisy
            "mix2": ndarray,        flip noisy
            "ref": ndarray,         clean
            "ref2": ndarray         flip clean
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["mix2"] = eg["mix2"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["ref2"] = eg["ref2"][s:s + self.chunk_size]
        
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["mix2"] = np.pad(eg["mix2"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["ref2"] = np.pad(eg["ref2"], (0, P), "constant")
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
