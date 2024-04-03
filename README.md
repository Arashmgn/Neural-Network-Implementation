## Neural network from scratch

A simple neural network implemented only using python and numpy

The first thing to do is to clone the repository:
```sh
$ git clone https://github.com/gomofficial/MLP-From-Scratch.git
```

to see the results of different tasks please view the NN from Scratch-Batch.ipynb notebook

- `--activation`: `relu` or `sigmoid` or `linear`
- `--learning_rate`: learning rate
- `--input-dims:int`: input dim
- `--layer-dims:list`: hidden/output layers dims


to create a new MLP neural network:
```python
$ mlp = MLP(4, layer_dims = [64, 64, 3], activationfuncs=[ "relu", "softmax"], learning_rate=0.001)
```