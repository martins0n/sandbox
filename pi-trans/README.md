#



## commands

```bash
python train.py --dataset-path='https://raw.githubusercontent.com/tinkoff-ai/etna/master/examples/data/example_dataset.csv' --n-epochs=300 --lr=0.005 --horizon=14

python train.py --dataset-path='pattern' --n-epochs=100 --lr=1e-4 --horizon=14 --n-layers=3 --d-ff=1024 --d-model=1024 --no-scale
```