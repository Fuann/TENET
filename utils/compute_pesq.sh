#!/bin/bash
# Copyright 2017 Johns Hopkins University (Author: Aswin Shanmugam Subramanian)
# Copyright 2021 NTNU (Author: Fuann)
# Apache 2.0

# This script creates the average PESQ score of files in an enhanced scp with corresponding
# files in a reference scp.
# Expects the PESQ third party executable in "utils/PESQ"
# PESQ source was dowloaded and compiled using "utils/download_se_eval_tool.sh"

set -e
set -u
set -o pipefail

if [ $# != 3 ]; then
   echo "Wrong #arguments"
   echo "Usage: utils/compute_pesq.sh <enhancement-scp> <source-scp> <expdir>"
   echo "Both scp must be sorted and comparable, output will be in expdir/pesq"
   exit 1;
fi

enhancement_scp=$1
source_scp=$2
expdir=$3
[ -d $expdir ] && mkdir -p $expdir
paste -d" " $source_scp $enhancement_scp | cut -d" " -f2,4 > $expdir/temp

echo "Compute PESQ (nb, wb)..."
n_files=$(wc -l < $expdir/temp)

# wb
t_wb_mos=0
avg_wb_mos=0

while read line; do

    ref=$(echo $line | cut -d" " -f1)
    enh=$(echo $line | cut -d" " -f2)

    # python - wb
    pesq_wb_score=`python -W ignore utils/pypesq.py $ref $enh`
    t_wb_mos=$(awk "BEGIN {print $t_wb_mos+$pesq_wb_score; exit}")
    #echo $$pesq_wb_score

done < $expdir/temp

# wb
avg_wb_mos=$(awk "BEGIN {print $t_wb_mos/$n_files; exit}")
echo $avg_wb_mos > "$expdir"/pesq-wb
rm $expdir/temp

echo "DONE"
