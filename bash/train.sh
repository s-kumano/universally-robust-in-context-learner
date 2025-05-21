#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){

  dname="batch_size=${batch_size}_d=${d}_N=${N}_lam=${lam}_eps=${eps}_epochs=${epochs}_lr=${lr}_seed=${seed}"
  f=${data_root}/train/${dname}

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 train.py \
      $batch_size \
      $d \
      $N \
      $lam \
      $eps \
      $epochs \
      $lr \
      $seed \
      $device \
      >> logs/${now}.out 2>&1
  fi
}

batch_size=1000
N=1000
lam=0.1
epochs=100
seed=0

d=20
lr=0.1
for eps in 0.0 0.098 0.95; do
  main
done

d=100
for eps_lr in "0.0 1.0" "0.06 0.2" "0.77 1.0"; do
  s=($eps_lr)
  eps=${s[0]}
  lr=${s[1]}
  main
done