import tensorflow as tf
import tflearn

from dataset import Dataset, Datasets

import pickle
import sys


# loading data
try:
    iris = pickle.load(open('iris.pkl', 'rb'))
    trainX, trainY, testX, testY = iris.load_data()
except:
    print("No dataset was found.")
    sys.exit(1)

# network parameters
input_dim = 4 # height data input
encoder_hidden_dim = 16
decoder_hidden_dim = 16
latent_dim = 4

# paths
TENSORBOARD_DIR='experiment/'
CHECKPOINT_PATH='out_models/'

# training parameters
n_epoch = 200
batch_size = 50


# encoder
def encode(input_x):
    encoder = tflearn.fully_connected(input_x, encoder_hidden_dim, activation='relu')
    mu_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear')
    logvar_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear')
    return mu_encoder, logvar_encoder

# decoder
def decode(z):
    decoder = tflearn.fully_connected(z, decoder_hidden_dim, activation='relu')
    x_hat = tflearn.fully_connected(decoder, input_dim, activation='linear')
    return x_hat

# sampler
def sample(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), dtype=tf.float32, name='epsilon')
    std_encoder = tf.exp(tf.mul(0.5, logvar))
    z = tf.add(mu, tf.mul(std_encoder, epsilon))
    return z

# loss function(regularization)
def calculate_regularization_loss(mu, logvar):
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), reduction_indices=1)
    return kl_divergence

# loss function(reconstruction)
def calculate_reconstruction_loss(x_hat, input_x):
    mse = tflearn.objectives.mean_square(x_hat, input_x)
    return mse

# trainer
def define_trainer(target, optimizer):
    trainop = tflearn.TrainOp(loss=target,
                              optimizer=optimizer,
                              batch_size=batch_size,
                              metric=None,
                              name='vae_trainer')

    trainer = tflearn.Trainer(train_ops=trainop,
                              tensorboard_dir=TENSORBOARD_DIR,
                              tensorboard_verbose=3,
                              checkpoint_path=CHECKPOINT_PATH,
                              max_checkpoints=1)
    return trainer


# flow of VAE training
def main():
    input_x = tflearn.input_data(shape=(None, input_dim), name='input_x')
    mu, logvar = encode(input_x)
    z = sample(mu, logvar)
    x_hat = decode(z)

    regularization_loss = calculate_regularization_loss(mu, logvar)
    reconstruction_loss = calculate_reconstruction_loss(x_hat, input_x)
    target = tf.reduce_mean(tf.add(regularization_loss, reconstruction_loss))

    optimizer = tflearn.optimizers.Adam().get_tensor()

    trainer = define_trainer(target, optimizer)

    trainer.fit(feed_dicts={input_x: trainX}, val_feed_dicts={input_x: testX},
                n_epoch=n_epoch,
                show_metric=False,
                snapshot_epoch=True,
                shuffle_all=True,
                run_id='VAE')

    return 0

if __name__ == '__main__':
    sys.exit(main())
