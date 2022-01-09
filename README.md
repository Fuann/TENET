# TENET: A Time-reversal Enhancement Network for noise-robust ASR (ASRU 2021)

A PyTorch implementation of our paper ["TENET: A Time-reversal Enhancement Network for noise-robust ASR"](https://arxiv.org/abs/2107.01531).
  
Audio samples can be found from [here](https://fuann.github.io/TENET).

## Requirements

See [requirements.txt](requirements.txt)

## Usage 
  
1. Download pretrained [wav2vec model](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models-1) and put it in `pretrain/wav2vec_large.pt`

2. Configure training settings and model hyperparameters from [nnet/conf.py](nnet/conf.py).  

3. Full experiment command:
``` yaml
Usage: ./train.sh <cpt-dir> <model> <exp-id>
  egs: ./train.sh exp/voicebank TENET(TCN-skip, DCCRN, PFPL, DPTNet) freqdpt_base
  options: --resume (path/to/best.pt.tar) 
           --gpu (0)
           --epochs (100)
           --batch-size (4) 
           --cache-size (10)
```

## Reference

This repository contains codes from:
* ConvTasNet - [https://github.com/funcwj/conv-tasnet](https://github.com/funcwj/conv-tasnet)
* DPTNet - [https://github.com/asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)
* DCCRN - [https://github.com/huyanxin/DeepComplexCRN](https://github.com/huyanxin/DeepComplexCRN)
* PFPL - [https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss](https://github.com/aleXiehta/PhoneFortifiedPerceptualLoss)
* SETK toolkit - [https://github.com/funcwj/setk](https://github.com/funcwj/setk)

## Citation

If you find this repository useful, please cite the following paper:

``` bibtex
@inproceedings{chao2021tenet,
  title = {TENET: A Time-reversal Enhancement Network for noise-robust ASR},
  author = {Fu-An Chao and Shao-Wei Fan Jiang and Bi-Cheng Yan 
            and Jeih-weih Hung and Berlin Chen},
  booktitle = {2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  year = {2021},
}
```


