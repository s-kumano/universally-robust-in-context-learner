#!/bin/bash

set -eux

data_root=data

mkdir -p logs $data_root

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

device=${@}

main(){

  dname="batch_size=${batch_size}_d_rob=${d_rob}_d_vul=${d_vul}_d_irr=${d_irr}_N_demos=${N_demos}_N_queries=${N_queries}_alpha=${alpha}_beta=${beta}_gamma=${gamma}_eps=${eps}_seed=${seed}"
  f=${data_root}/test/${weight_name}/${dname}

  echo $f >> logs/${now}.out 2>&1

  if [ ! -e $f ]; then
    python3 test.py \
      $weight_name \
      $eps \
      $seed \
      $device \
      normal \
      $batch_size \
      $d_rob \
      $d_vul \
      $d_irr \
      $N_demos \
      $N_queries \
      $alpha \
      $beta \
      $gamma \
      >> logs/${now}.out 2>&1
  fi
}

eps=0.2
batch_size=1000
d_rob=10
d_vul=90
d_irr=0
N_demos=1000
N_queries=1000
alpha=1.0
beta=0.1
gamma=0.1
seed=0

for weight_name in ideal_std ideal_adv; do

for eps in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  main
done
eps=0.2

for d_rob in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30; do
  main
done
d_rob=10

for d_vul in 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000; do
  main
done

d_vul=0
for d_irr in 10 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000; do
  main
done
d_vul=90
d_irr=0

done