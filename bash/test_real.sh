#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){

  if [ $N_demos ]; then
    dname="dataset_name=${dataset_name}_N_demos=${N_demos}_eps=${eps}_seed=${seed}"
    N_demos="--N_demos ${N_demos}"
  else
    dname="dataset_name=${dataset_name}_N_demos=None_eps=${eps}_seed=${seed}"
    N_demos=""
  fi

  f=${data_root}/test/${weight_name}/${dname}

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 test.py \
      $weight_name \
      $eps \
      $seed \
      $device \
      real \
      $dataset_name \
      $N_demos \
      >> logs/${now}.out 2>&1
  fi
}

seed=0

for weight_name in ideal_std ideal_adv; do

for dataset_name in MNIST FMNIST CIFAR10; do

N_demos=""
for eps in 0.0 0.05 0.1 0.15 0.2 0.25 0.3; do
  main
done

for eps in 0.0 0.1; do
for N_demos in 1 5 10 50 100; do
  main
done
done

done

dataset_name=MNIST
for eps in 0.0 0.1; do
for N_demos in 500 1000; do
  main
done
done

done