export CUDA_VISIBLE_DEVICES=0,1,2,3
i=0
python ./run_ner.py \
--model_name_or_path /home/zju/lyl/bert_base_zh \
--model_type Bert \
--task_type Military_NER \
--num_labels 9 \
--test_dir ../data/ \
--do_test \
--index 0 \
--stacking False \
--data_dir ../data/data/data_$i \
--output_dir ../output/eval_results_bert_base_zh_360_1e-05_32_1000_Military_NER_0-fold/ \
--max_seq_length 360 \
--max_question_length 0 \
--per_gpu_eval_batch_size 8 \
