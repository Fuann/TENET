#!/usr/bin/env bash

set -eu
stage=1
eval_stage=1    # 3: make sep.scp,  # 4: compute sisnr,  # 5: compute pesq

# Check nnet/conf.py first !
# exp config
epochs=100

# constrainted by GPU number & memory
gpu=0
batch_size=4    #4
cache_size=10   #10
resume=""
resume_opt=""

# testing data
dataset="voicebank"
test="tt tt-qut"

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 <cpt_dir> <model(TCN-skip, DCCRN, PFPL, DPTNet, TENET)> <exp-id>"
    echo "  egs: $0 exp/voicebank TENET freqdpt_base"
    echo "  options: --resume (path/to/best.pt.tar)"
    echo "           --gpu (0) --epochs (100) --batch-size (4) --cache-size (10)"
    exit 1
fi

cpt_dir=$1
model=$2
expid=$3

mkdir -p $cpt_dir/$model


[ ! -z $resume ]  && resume_opt="--resume $resume"

if [ $stage -le 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu \
    python nnet/train.py $resume_opt \
      --model $model \
      --gpus $gpu \
      --epochs $epochs \
      --batch-size $batch_size \
      --num-workers $cache_size \
      --checkpoint $cpt_dir/$model/$expid
      #> $cpt_dir/$model/$expid.train.log
fi

if [ $stage -le 2 ]; then

    for i in $test; do
        clean_scp=data/$dataset/$i/clean.scp
        noisy_scp=data/$dataset/$i/noisy.scp

        CUDA_VISIBLE_DEVICES=0 \
        ./inference.sh --stage $eval_stage \
            --model $model \
            --gpu 0 \
            --ref-scp $clean_scp \
            $noisy_scp  \
            $cpt_dir/$model/$expid \
            $cpt_dir/$model/$expid/$test
    done
fi

./result.sh $dataset
