python finetune.py \
    --model_name_or_path sberbank-ai/rugpt3small_based_on_gpt2 \
    --train_file ../data/datasets/kashin-v1-train.json \
    --validation_file ../data/datasets/kashin-v1-test.json \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ../tmp/test-clm