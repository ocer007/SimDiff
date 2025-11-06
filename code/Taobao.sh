#!/bin/bash

# 参数赋值（Taobao, 2025-09-10 20:08:06）
text_dim=4096
batch_size=1024
hidden_factor=64
timesteps=350
beta_end=0.02
beta_start=0.0001
lr=0.0059257503525119055
l2_decay=0.0009824816048528869
dropout_rate=0.0341222811345696
w=2.0
p=0.1
optimizer='adamw'
n_layers=2
statesize=20
tstBat=9182
tstEpoch=1
data='Taobao'
data_dir='./data'
layers=1
flag=2
bpr_alpha=0.6086591370027884
reg_alpha=0.00848840369257049
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
