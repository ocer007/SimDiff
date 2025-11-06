#!/bin/bash

text_dim=768
batch_size=2000
hidden_factor=64
timesteps=200
beta_end=0.02
beta_start=0.0001
lr=0.03
l2_decay=0.0
dropout_rate=0.1
w=2.0
p=0.1
optimizer='adam'
n_layers=2
statesize=10
tstBat=2000
tstEpoch=1
data='office'
data_dir='./data'
layers=1
flag=2
bpr_alpha=0.9
reg_alpha=0.003
cuda=0
random_seed=2024
name='test'

echo "Generated name: $name"

python main.py \
  --text_dim $text_dim --batch_size $batch_size \
  --hidden_factor $hidden_factor --timesteps $timesteps --beta_end $beta_end \
  --beta_start $beta_start --lr $lr --l2_decay $l2_decay --dropout_rate $dropout_rate \
  --w $w --p $p --optimizer $optimizer --n_layers $n_layers --statesize $statesize \
  --tstBat $tstBat --tstEpoch $tstEpoch --data $data --data_dir $data_dir \
  --layers $layers \
  --flag $flag --reg_alpha $reg_alpha --bpr_alpha $bpr_alpha \
  --cuda $cuda --random_seed $random_seed --name "$name"
