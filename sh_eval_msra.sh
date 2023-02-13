# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# for step in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000
for step in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000 22000 24000 26000 28000 30000 32000 34000 36000 38000 40000 42000 44000 46000 48000 50000 52000 54000
do
	echo "Run evaluate msra on steps $step"
	python run_eval.py \
	--dev_dir ./ner_datasets/MSRA_test.json \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_width 10 \
	--model /mnt/lustrenew/lujinghui1/Unified_NER/my_trained_models_10000/$step/ \
	--dataset_name msra \
	--eval_result_dir ./unified_ner_v1.5/results_$step/ \
	--batch_size 4
done
# spring.submit arun --gpu -n1 -s --job-name=R-SC210077.00110 "bash cluster_run.sh"