import tensorflow as tf
import tflearn

import sys


# paths
TENSORBOARD_DIR='experiment/'
CHECKPOINT_PATH='out_models/'

class VAE(object):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    # encoder
    def encode(self, input_x):
        encoder = tflearn.fully_connected(input_x, 16, activation='relu')
        mu_encoder = tflearn.fully_connected(encoder, self.latent_dim, activation='linear')
        logvar_encoder = tflearn.fully_connected(encoder, self.latent_dim, activation='linear')
        return mu_encoder, logvar_encoder

    # decoder
    def decode(self, z):
        decoder = tflearn.fully_connected(z, 16, activation='relu')
        x_hat = tflearn.fully_connected(decoder, self.input_dim, activation='linear')
        return x_hat

    # sampler
    def sample(self, mu, logvar):
        epsilon = tf.random_normal(tf.shape(logvar), dtype=tf.float32, name='epsilon')
        std_encoder = tf.exp(tf.mul(0.5, logvar))
        z = tf.add(mu, tf.mul(std_encoder, epsilon))
        return z

    # loss function(regularization)
    def calculate_regularization_loss(self, mu, logvar):
        kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), reduction_indices=1)
        return kl_divergence

    # loss function(reconstruction)
    def calculate_reconstruction_loss(self, x_hat, input_x):
        mse = tflearn.objectives.mean_square(x_hat, input_x)
        return mse

    # trainer generator
    def return_trainer(self, input_x, optimizer, batch_size):
        # encode
        self.mu, self.logvar = self.encode(input_x)
        # sampling
        z = self.sample(self.mu, self.logvar)
        # decode
        self.x_hat = self.decode(z)

        # calculate loss
        regularization_loss = self.calculate_regularization_loss(self.mu, self.logvar)
        reconstruction_loss = self.calculate_reconstruction_loss(self.x_hat, input_x)
        target = tf.reduce_mean(tf.add(regularization_loss, reconstruction_loss))

        # define trainer
        trainop = tflearn.TrainOp(loss=target,
                                  optimizer=optimizer,
                                  batch_size=batch_size,
                                  name='vae_trainer')

        trainer = tflearn.Trainer(train_ops=trainop,
                                  tensorboard_dir=TENSORBOARD_DIR,
                                  tensorboard_verbose=3,
                                  checkpoint_path=CHECKPOINT_PATH,
                                  max_checkpoints=1)
        return trainer

    def return_evaluator(self, trainer):
        evaluator = tflearn.Evaluator([self.mu, self.logvar], session=trainer.session)
        return evaluator

    # loading checkpoint
    def get_checkpoint(self):
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_PATH)
        if ckpt:
            last_model = ckpt.model_checkpoint_path
            return last_model
        else:
            print("No trained model was found.")
            sys.exit(0)
