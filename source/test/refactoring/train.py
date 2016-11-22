import tensorflow as tf
import tflearn

from VAE import VAE

from dataset import Dataset, Datasets

import pickle
import sys


# load data
iris = pickle.load(open('iris.pkl', 'rb'))
trainX, trainY, testX, testY = iris.load_data()

# define input and output
input_dim = 4
latent_dim = 4

# define training parameters
n_epoch = 200
batch_size = 50

# flow of VAE training
def main():
    vae = VAE(input_dim, latent_dim)

    input_x = tflearn.input_data(shape=(None, input_dim), name='input_x')
    optimizer = tflearn.optimizers.Adam().get_tensor()

    trainer = vae.return_trainer(input_x, optimizer, batch_size)

    trainer.fit(feed_dicts={input_x: trainX}, val_feed_dicts={input_x: testX},
                n_epoch=n_epoch,
                shuffle_all=True,
                run_id='VAE')


# entry point
if __name__ == '__main__':
    sys.exit(main())
