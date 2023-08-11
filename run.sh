#!/bin/zsh

nohup mpirun -np 5 python ./ceval.py >& ceval.0801.log &
