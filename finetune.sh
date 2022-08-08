#! /bin/bash

poetry run python src/finetune.py \
    --train_path examples/livedoor/tmp/train.csv \
    --valid_path examples/livedoor/tmp/valid.csv \
    --model_name_or_path cl-tohoku/bert-base-japanese-v2 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --output_dir tmp/