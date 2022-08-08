# Example Code for Text Classification with Transformers

## Usage
### Build a Docker image 
```sh
export TAG=python/cuda11.3-cudnn8:dev
export STAGE=development
docker build . --tag $TAG --target $STAGE
```

### Finetune the japanese text classifier
```sh
WS=$(pwd)

mkdir -p examples/livedoor/tmp && cd "$_"
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz -O - | tar -xvf
python livedoor.py

cd $WS
python src/finetune.py \
    --train_path examples/livedoor/tmp/train.csv \
    --valid_path examples/livedoor/tmp/valid.csv \
    --output_dir ./tmp/ \
    --model_name_or_path cl-tohoku/bert-base-japanese-v2 \
    --num_train_epochs 5
```