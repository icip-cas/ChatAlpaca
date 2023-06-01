GPUS_PER_NODE=8

torchrun --nproc_per_node=8 --master_port=20001 train/train_hf.py \
    --model_name_or_path /path/to/llama-7b  \
    --data_path /path/to/chatalpaca-20k-data \
    --bf16 True \
    --output_dir /path/to/chatalpaca-7b-hf \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True