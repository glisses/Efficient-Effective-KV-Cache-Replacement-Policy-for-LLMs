python exampels/eval_long_ppl.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name pg19 --split test \
--enable_start_recent_kv_cache --enable_pos_shift --start_size 4 --recent_size 2048