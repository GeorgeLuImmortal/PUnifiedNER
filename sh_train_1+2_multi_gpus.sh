# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_unified_ner_multi_gpus.py \
--train_batch_size_per_gpu 8 \
--dev_batch_size_per_gpu 4 \
--lr 1e-4 \
--epoch 100 \
--max_source_length 512 \
--max_target_length 512 \
--method 1+2 \
--save_dir /mnt/lustre/lujinghui1/Unified_NER/my_trained_models_method_1+2/ \
--model /mnt/lustre/lujinghui1/Unified_NER/pretrained_ckpt_1e-4/6000/ \
--train_dir ./ner_datasets/ml_train.json  \
--dev_dir ./ner_datasets/ml_test_all.json \
--eval_steps 100
# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"