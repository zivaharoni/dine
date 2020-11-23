#!/usr/bin/env bash
date
cd /common_space_docker/ziv/dine || exit
for i in {1..2}
do
  CUDA_VISIBLE_DEVICES=0, python ./main.py --config ./configs/synthetic.json --exp_name di_estimate --data_subname awgn &
done
sleep 120