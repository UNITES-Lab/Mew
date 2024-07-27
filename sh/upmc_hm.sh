device=4

data=upmc
task=cox
shared=True
attn_weight=True
lr=0.001
num_layers=1
emb_dim=256
batch_size=16
drop_ratio=0.0

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
