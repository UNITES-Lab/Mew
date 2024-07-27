device=0

data=charville
task=classification
shared=False
attn_weight=False
lr=0.0001
num_layers=4
emb_dim=512
batch_size=16
drop_ratio=0.5

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
