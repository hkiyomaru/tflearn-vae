# TFLearn implementation of Variational Autoencoder

## Development Environment

* OS: Ubuntu14.04
* Python: 3.4.5
* TensorFlow: 0.10.0
* TFLearn: 0.2.1

## Install dependent packages

I recommend using pyenv and venv.

```
$ pyenv install 3.4.5
$ pyenv local 3.4.5
$ pyvenv venv
$ . venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip3 install -r requirements.txt
```

When you'd like to deactivate the environment,

```
(venv) $ deactivate
```

## Install TensorFlow

Then, install TensorFlow.

TFLearn doesn't run on TensorFlow0.11 yet,
so you have to install the earlier version.

```
# Ubuntu/Linux 64-bit, CPU only, Python 3.4
(venv) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl

# Ubuntu/Linux 64-bit, GPU enabled, Python 3.4
# Requires CUDA toolkit 7.5 and CuDNN v5. For other versions, see "Install from sources" below.
(venv) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0-cp34-cp34m-linux_x86_64.whl

# Mac OS X, CPU only, Python 3.4 or 3.5:
(venv) $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl
```

```
(venv) $ pip3 install --upgrade $TF_BINARY_URL
```

If you have trouble, click [here](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html) for detail.

## Create dataset

Move to `source` directory.

First, run this script to create a dataset.

```
(venv) $ python dataset.py
```

Make sure that there is `h_and_w.pkl` in the current directory.

This dataset has 1000 samples and each sample has 2-dimensional feature and its class label.

This is the simulation of human height and weight.

## Train

Then, train VAE.

```
(venv) $ python vae.py
```

Check the loss-transition using tensorboard.

```
(venv) $ tensorboard --logdir=<absolute path to current directory>/experiment
```

Open your browser, and get access to `localhost:6006`.

You can show the loss-transition and the computational graph.

## Classify

Extract encoder model and make its outputs features of samples.

The features represent the distributions that generate the original samples.

Now, classify the samples using SVM.

```
(venv) $ python classify.py
```
