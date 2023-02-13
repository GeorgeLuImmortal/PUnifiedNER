# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# for step in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000

echo "Run evaluate cluener"
python run_test_cluener.py \
--version 1.5 \
--model /mnt/lustrenew/lujinghui1/Unified_NER/my_trained_models_10000/58000/ \
--batch_size 8 \
--dev_dir ./ner_datasets/cluener_test.json \
--max_target_length 512 \
--max_source_length 512

# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"