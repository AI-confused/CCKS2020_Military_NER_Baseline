export CUDA_VISIBLE_DEVICES=0,1,2,3
i=0
python ./run_ner.py \
--model_name_or_path /home/zju/lyl/bert_base_zh \
--model_type bert \
--task_type Military_NER \
--FGM \
--do_train \
--do_eval \
--do_test \
--index 0 \
--stacking False \
--num_labels 9 \
--data_dir ../data/data/data_$i \
--test_dir ../data/ \
--output_dir ../output/ \
--max_seq_length 360 \
--max_question_length 0 \
--eval_steps 5 \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 20 \
--learning_rate 1e-5 \
--train_steps 1500