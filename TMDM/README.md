激活虚拟环境
conda activate TMDM

进入当下目录

cd /root/TMDM_MOM


运行命令：
python runner9_NS_transformer.py \
  --data custom \
  --is_training True \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --features M \
  --target OT \
  --freq d \
  --enc_in 8 --dec_in 8 --c_out 8 \
  --model_id TMDM-r-MOM1 \
  --timesteps 200 \
  --test_batch_size 4 \
  --simpatch_d_model 128 \
  --simpatch_layers 1 \
  --simpatch_heads 4 \
  --simpatch_d_ff 256 \
  --sample_temperature 1.5 \
  --point_agg trimmed_mean \
  --trim_ratio 0.2 \
  --seed 2022
