CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--model_name bert_spc \
--dataset  sem16 \
--learning_rate 5e-5 \
--max_seq_len 100 \
--num_epoch 50 \
--tabsa \
--bert_dim 1024 \
--batch_size 8 \
#--classifier
