#! /bin/bash

poetry run python src/evalute.py \
    --test_path examples/livedoor/tmp/test.csv \
    --save_items_dir tmp/ \
    --evaluation_strategy epoch \
    --output_dir tmp/predict/