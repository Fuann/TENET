#!/bin/bash
# This script is local version

set -eu

# constrainted by GPU number & memory
model='TENET'
gpu=0
fs=16000
nj=1

st=
ed=

stage=1

ref_scp=
remove_wav=false

. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
    echo "Usage: $0 <noisy.scp> <cpt-dir> <dump-dir>"
    echo "  option:  --ref-scp(given clean counterpart and calculate sisnr) --remove-wav(false)"
    echo "           --model(TENET) --gpu GPUID(0) --fs sample-rate(16000) --nj (1)"
    exit 1
fi

noisy_scp=$1
cptdir=$2
dumpdir=$3

logdir=$3/logdir
[ -d $logdir ] && rm -r $logdir

## usage: separate.py [-h] --input INPUT [--gpu GPU] [--fs FS] [--dump-dir DUMP_DIR]
##                  checkpoint

if [ $stage -le 1 ]; then
    echo "Spliting scripts for $nj ..."
    mkdir -p $logdir

    split_scp=""
    for i in `seq $nj`; do
        split_scp="$split_scp $logdir/wav.$i.scp"
    done
    utils/split_scp.pl $noisy_scp $split_scp || exit 1;
fi

if [ $stage -le 2 ]; then

    [ -z $st ] && st=1
    [ -z $ed ] && ed=$nj

    for i in `seq $st $ed`; do
    echo "Process $logdir/wav.$i.scp (total $nj)"
        python3.6 nnet/separate.py \
            --model $model \
            --gpu $gpu \
            --fs $fs \
            --input $logdir/wav.$i.scp \
            --dump-dir $dumpdir \
            $cptdir
    done

fi

if [ $stage -le 3 ]; then
    echo "Creating separates sep.scp ..."
    enh_wav_dir=$dumpdir/enhance
    cut -f1 -d" " $noisy_scp > $dumpdir/id
    awk '{print $1}' $noisy_scp | sed s:^:$PWD/$enh_wav_dir/:g | sed s:$:.wav:g > $dumpdir/path
    paste -d" " $dumpdir/id $dumpdir/path > $dumpdir/sep.scp
    rm $dumpdir/id $dumpdir/path
fi

[ -z $ref_scp ] && exit 0

if [ $stage -le 4 ]; then
    echo "Compute SI-SNR to $dumpdir/sisnr"
    python3 sptk/compute_si_snr.py \
        $dumpdir/sep.scp $ref_scp > $dumpdir/sisnr
fi

if [ $stage -le 5 ]; then
    echo "Compute PESQ to $dumpdir/pesq"
    utils/compute_pesq.sh \
        $dumpdir/sep.scp $ref_scp $dumpdir
fi

[ "$remove_wav" == "true" ] && rm -r $dumpdir/enhance

exit 0
