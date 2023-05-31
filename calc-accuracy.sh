#!/bin/zsh
grep Task yiyan-ceval-result-rank* | awk '{ print $3, $4; }' | sed 's/correct=//;s/total=//;' | awk -F',' '{ corr+= $1; total+=$2; }END{ print corr, total, corr/total}'
