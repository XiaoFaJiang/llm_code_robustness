conda activate adv
export CUDA_VISIBLE_DEVICES=10,11,12,13
python attack.py \
    --csv_store_path replace1.csv \
    --language=python \
    --model_name_or_path=/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B \
    --eval_batch_size 4 \
    --use_sa \
    --beam_size 1 \
    --transfrom_iters 2 \
    --perturbation_type=rename \
    --seed 42| tee replace1.log