fs = 16000
chunk_len = 4  # 4 (s)
chunk_size = chunk_len * fs
num_spks = 2

# network configure
# TCN-skip (ConvTasNet)
skip_conf = {
    "L": 16,
    "N": 256,
    "X": 8,
    "R": 3,
    "B": 128,
    "H": 256,
    "S": 128,
    "P": 3,
    "norm": "gLN",
    "num_spks": num_spks,
    "non_linear": "relu",
    "causal": False,
}

# DCCRN
dccrn_conf = {
    # dccrn-cl conf
    "rnn_units": 256,
    "masking_mode": 'E',
    "use_clstm": True,
    "kernel_num": [32, 64, 128, 256, 256, 256]
}

# DCUNet (for PFPL)
dcunet_conf = {
    # dcunet-20 conf
    "hidden_size": 257,
}

# DPTNet (time)
# CDPT (freq)
# CD-DPTNet (merge)
dpt_conf = {
    "feat_type": "freq",    # time, freq, merge
    "n_filters": 512,      # 256, 512, 256
    "kernel_size": 400,    # 16, 400, 16
    "kernel_shift": 100,   # 8, 100, 8
    "n_feats": 128,        # 64, 128, 64
    "n_src": 2,
    "n_heads": 8,       # 8, 8, 8
    "ff_hid": 256,      # 128, 256, 128
    "mha_activation": "relu",
    "ff_activation": "relu",
    "chunk_size": 100,   # 100
    "hop_size": None,
    "n_repeats": 5,     # 5, 5, 5
    "n_intra": 1,
    "n_inter": 1,
    "norm_type": "gLN",
    "mask_act": "tanh",   # "relu", "tanh", None
    "mask_floor": 0.0,
    "bidirectional": True,
    "dropout": 0.0,     #0.1 when ff_hid=256 up
    "sparse_topk": None, # None
}

# TENET conf
tenet_conf = {
    # Choose a model, and config them above
    "model": "DPTNet",
}

# data configure:
## voicebank ## no noise.scp
train_dir = "data/voicebank/tr/"
dev_dir = "data/voicebank/cv/"

train_data = {
    "mix_scp":
    train_dir + "noisy.scp",
    "ref_scp":
    train_dir + "clean.scp",
    "sample_rate":
    fs,
    "flip":             # time-reversal
    True,
    "speed_perturb":    # speed perturb
    True,
    "mask":             # sample masking
    True,
    "shift":            # time shifting
    True,
    "reverb":           # add reverb
    False,
}

dev_data = {
    "mix_scp":
    dev_dir + "noisy.scp",
    "ref_scp":
    dev_dir + "clean.scp",
    "sample_rate":
    fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-3,                #1e-3,
    "weight_decay": 1e-5,    #1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "optimizer_swa": True,
    "clip_norm": 5,         # 5 for DPRNN/DPTNet model else 0
    "min_lr": 1e-8,
    "patience": 2,          # 2
    "no_impr": 8,           # 8
    "factor": 0.5,
    "logging_period": 200,  # batch number
    "model_type": "wav2vec",    # tenet & pfpl
    "loss_type": "lp",          # tenet & pfpl
    "pretrained_model_path": "pretrain/wav2vec_large.pt",   # tenet & pfpl
    "alpha": 2000,  # tenet
    "beta": 0.7,    # tenet
    "gamma": 0.3    # tenet
}
