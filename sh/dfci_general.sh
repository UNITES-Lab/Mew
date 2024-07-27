device=2

data=dfci
task=classification
shared=False
attn_weight=True
lr=0.001
num_layers=2
emb_dim=128
batch_size=16
drop_ratio=0.25

python main.py \
  --data $data \
  --task $task \
  --shared $shared \
  --attn_weight $attn_weight \
  --lr $lr \
  --num_layers $num_layers \
  --emb_dim $emb_dim \
  --batch_size $batch_size \
  --drop_ratio $drop_ratio \
  --device $device 
