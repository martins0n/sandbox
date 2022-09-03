python finetune.py \
    --model_name_or_path sberbank-ai/rugpt3medium_based_on_gpt2 \
    --train_file ../data/datasets/kashin-v1-train.json \
    --validation_file ../data/datasets/kashin-v1-test.json \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --output_dir ../tmp/test-clm \
    --save_strategy epoch \
    --logging_strategy epoch \
    --num_train_epochs 5