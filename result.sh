#!/bin/bash
# fuann@2020

echo == test-demand  ==
for x in exp/voicebank*/*/*/tt; do
    sisnr=$(cat $x/sisnr | awk 'NR==2 {print}' | cut -d'/' -f2)
    pesq_wb=$(cat $x/pesq-wb)
    echo "SISNR:" $sisnr "  PESQ-WB:" $pesq_wb "  " $x
    #pesq_nb=$(cat $x/pesq)
    #echo "SISNR:" $sisnr "  PESQ-NB:" $pesq_nb "  PESQ-WB:" $pesq_wb "  " $x
done | sort -nrk1

echo == test-qut  ==
for x in exp/voicebank*/*/*/tt-qut; do
    sisnr=$(cat $x/sisnr | awk 'NR==2 {print}' | cut -d'/' -f2)
    pesq_wb=$(cat $x/pesq-wb)
    echo "SISNR:" $sisnr "  PESQ-WB:" $pesq_wb "  "$x
done | sort -nrk2