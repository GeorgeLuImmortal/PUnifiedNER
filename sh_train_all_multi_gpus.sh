# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_all_multi_gpus.py \
--train_batch_size_per_gpu 4 \
--dev_batch_size_per_gpu 8 \
--lr 1e-4 \
--steps 1000000 \
--start_step 0 \
--warm_up_step 5000 \
--max_source_length 512 \
--max_target_length 512 \
--method 3 \
--save_dir /mnt/lustrenew/lujinghui1/Unified_NER/my_trained_models_10000/ \
--model /mnt/lustrenew/lujinghui1/Unified_NER/pretrained_ckpt_1e-4/10000/ \
--tokenizer ./models/my_t5_base/ \
--data_dir ./ner_datasets/  \
--eval_steps 2000 \
--beam_width 5 \
--decode_max_len 512 \
--model_max_len 512 \
--num_gpus 8 \
--max_entities 35 \
--random_seed 0
# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"