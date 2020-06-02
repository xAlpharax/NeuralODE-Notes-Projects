# Neural Ordinary Differential Equations

Repository for notes, snippets, and projects on NODEs. Includes results after training CNN based networks with different methods on MNIST and CIFAR-10 datasets accordingly.

## Introduction

Neural ODEs are a method of extending [Residual Neural Networks](https://arxiv.org/abs/1512.03385) for smoother gradients and general improvements in training. The original paper [(2018) Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) and later on [ANODE](https://arxiv.org/abs/1902.10298) will be papers cited for their techniques and algorithms.

In addition, [Neural ODEs for undergraduate students](https://drive.google.com/file/d/13uynuOgbnbAjmbHWo8-DbS-0dlKtdF4T/view?usp=sharing) has a great introduction into the subject of matter while [Towards Understanding Normalization in Neural ODEs](https://arxiv.org/abs/2004.09222) explains new advancements in Normalizing NODEs.

## Installation

```bash
#cloning the repo
git clone https://github.com/xAlpharax/NeuralODE-Notes-Projects

#install dependencies
pip3 install -q -r requirements.txt
```

## Training

Training the network created in dcodnn.py can be done by running the following command. Note that the weights and visuals are placed in their separate directories respectively.
```bash
#training DCODNN for 30 epochs with the given parameters (CIFAR)
python3 train-node/train-cifar-10.py

#training lighter DCODNN for 5 epochs with the given parameters (MNIST)
python3 train-node/train-mnist.py
```

Moreover, training residual networks with similar structure as the ones above is done by:
```bash
#training DRCNN for 30 epochs with the given parameters (CIFAR)
python3 train-res/train-res-cifar-10.py

#training lighter DRCNN for 5 epochs with the given parameters (MNIST)
python3 train-res/train-res-mnist.py
```

## Results and comparison

|             | CIFAR               | MNIST               |
| ----------- |:-------------------:| --------------------|
| DCODNN      | val_acc: 0.7059     | **val_acc: 0.9865** |
| DRCNN       | **val_acc: 0.7545** | val_acc: 0.9824     |

## Contributing
In case someone would like this, pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
