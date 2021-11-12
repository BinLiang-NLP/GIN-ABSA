CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
--model_name gin \
--dataset  sem16 \
--learning_rate 1.2e-3 \
--max_seq_len 100 \
--num_epoch 30 \
--tabsa \
--tabsa_with_absa \
--batch_size 16 \
--seed 19
#--classifier
