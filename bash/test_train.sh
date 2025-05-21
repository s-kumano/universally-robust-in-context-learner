#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){

  dname="batch_size=${batch_size}_d=${d}_N_demos=${N_demos}_N_queries=${N_queries}_lam=${lam}_eps=${eps}_seed=${seed}"
  f=${data_root}/test/${weight_name}/${dname}

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 test.py \
      $weight_name \
      $eps \
      $seed \
      $device \
      train \
      $batch_size \
      $d \
      $N_demos \
      $N_queries \
      $lam \
      >> logs/${now}.out 2>&1
  fi
}

eps=0.15
batch_size=1000
d=100
N_demos=1000
N_queries=1000
lam=0.1
seed=0

for weight_name in ideal_std ideal_adv; do

main

for eps in 0.0 0.05 0.06 0.07 0.2 0.21 0.22; do
  main
done
eps=0.15

for d in 5 10 20 160 170 180; do
  main
done
d=100

done