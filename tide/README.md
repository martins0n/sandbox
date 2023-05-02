## start

> python pipe.py +dataset=pattern +model=pattern  model.max_epochs=100
> python pipe.py +dataset=electricity +model=electricity_96  model.max_epochs=50 accelerator=cuda model.train_batch_size=1024 model.test_batch_size=1024 model.lr=0.0009

ToDo:
* Add cosine scheduler