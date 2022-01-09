import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2VecModel
#import os
#from fairseq import checkpoint_utils
#from fairseq.checkpoint_utils import load_model_ensemble_and_task
from functools import partial

class w2v_encoder(nn.Module):
    def __init__(self, 
        PRETRAINED_MODEL_PATH = '/path/to/wav2vec_large.pt'
    ):
        super(w2v_encoder, self).__init__()
        ckpt = torch.load(PRETRAINED_MODEL_PATH)
        self.model = Wav2VecModel.build_model(ckpt['args'], task=None)
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.feature_extractor
        self.model = nn.DataParallel(self.model)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    enc = w2v_encoder(PRETRAINED_MODEL_PATH="/share/nas167/fuann/pretrain/wav2vec_large.pt")
    x = torch.rand((1, 64000))
    out = enc(x)
    print(out.shape)