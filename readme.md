# This is the official code repository for the Neurips 2025 submission: "Attention Learning is Needed to Efficiently Learn Parity Functions"
## 1. training transformer for k-parity
please use training_script.py to train single-layer, k attention heads transformer for parity. The configs you can have for this script include:

- number_of_samples: the number of samples you want to specify to train the transformer, if you don't specify, a number approximating the full-batch will be used (2 ** length * 0.8)
- device: the device on which you want the training to happen, default will be cuda if you have a gpu
- length: the input length of the sequence, default is 16 because 20 takes a long time to train
- k: the size of the parity bits, default number being 3
- batch_size: the batch size for the dataloader, SGD is used and the default batch size is 80000.
- arly_stop whether to use early stopping, which is recommend to leave as default, which is False, because loss could drop further in later epochs even after some plateau.
- learning_rate: the configuraion i used to train the network and generate the visualisation is default learning rate.
- patience: won't be used if you don't specify early_stop
- min_delta: won't be used if you don't specify early_stop
- epochs: how many number of epochs you want to train, the default value being 2000.
- visualize_attention whether you want to save the attention heatmap, default is True
- save_loss: whether you want to save the loss trend, default is also True
- log_dir: the directory to save your losses and plots.

for example, if you want to run the configuration, n=20 and k=4 without plotting the visualization (maybe you want to do that), just run the script with:

```python training_script.py --length 20 --k 4 --visualize_attention False```

## 2. running numerical experiments for sparse linear regression and sparse gene classification
follow the instructions sparse_regression.ipynb and gene_classification.ipynb, run block by block and visualize your own loss trend with different configurations!