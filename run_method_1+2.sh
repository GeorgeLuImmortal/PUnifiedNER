# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train_unified_ner.py \
--batch_size 16 \
--lr 1e-5 \
--epoch 75 \
--max_source_length 256 \
--max_target_length 256 \
--method 1+2 \
--save_dir /mnt/lustrenew/lujinghui1/Unified_NER/my_trained_models_method_1+2/ \
--eval_steps 1200 \
--train_dir /mnt/lustrenew/lujinghui1/dml_ner/dataset/cluener/ml_train.json \
--dev_dir /mnt/lustrenew/lujinghui1/dml_ner/dataset/cluener/ml_test_all.json \
--model /mnt/lustrenew/lujinghui1/Unified_NER/models/my_t5_base/ \
--tokenizer /mnt/lustrenew/lujinghui1/Unified_NER/models/my_t5_base/ \
--warm_up_step 6000
# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"