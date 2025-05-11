# Question - Answering Using BART

## Train
```bash
   !python -m main --data_path=/kaggle/working/data/train_data_name.pkl --checkpoint_interval=100
```

## Inference
```bash
   !python -m modules.inference 
```

!python -m main --data_path=/kaggle/input/synthetic-personachat/valid.pkl --validation_path=/kaggle/input/synthetic-personachat/valid.pkl --learning_rate=1e-5


