import numpy as np

import tensorflow as tf
import tflearn

from dataset import Dataset, Datasets

import pickle
import sys


# loading data
try:
    height = pickle.load(open('h_and_w.pkl', 'rb'))
    trainX, trainY, testX, testY = height.load_data()
except:
    print("No dataset was found.")
    sys.exit(1)

# network parameters
input_dim = 1 # height data input
encoder_hidden_dim = 16
decoder_hidden_dim = 16
latent_dim = 2

# paths
TENSORBOARD_DIR='experiment/'
CHECKPOINT_PATH='out_models/'

# training parameters
batch_size = 50


# encoder
def encode(input_x):
    encoder = tflearn.fully_connected(input_x, encoder_hidden_dim, activation='relu')
    mu_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear')
    logvar_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear')
    return mu_encoder, logvar_encoder

# decoder
def decode(z):
    decoder = tflearn.fully_connected(z, decoder_hidden_dim, activation='relu', restore=False)
    x_hat = tflearn.fully_connected(decoder, input_dim, activation='linear', restore=False)
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
                              tensorboard_verbose=0,
                              checkpoint_path=CHECKPOINT_PATH,
                              max_checkpoints=1)
    return trainer

# evaluator
def define_evaluator(trainer, mu, logvar):
    evaluator = tflearn.Evaluator([mu, logvar], session=trainer.session)
    return evaluator

# loading checkpoint
def get_checkpoint(out_models_dir):
    ckpt = tf.train.get_checkpoint_state(out_models_dir)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        return last_model
    else:
        print("No trained model was found.")
        sys.exit(0)

# flow of SVM classification
def main():
    global trainX, trainY, testX, testY

    input_x = tflearn.input_data(shape=(None, input_dim), name='input_x')
    mu, logvar = encode(input_x)
    z = sample(mu, logvar)
    x_hat = decode(z)

    regularization_loss = calculate_regularization_loss(mu, logvar)
    reconstruction_loss = calculate_reconstruction_loss(x_hat, input_x)
    target = tf.reduce_mean(tf.add(regularization_loss, reconstruction_loss))

    optimizer = tflearn.optimizers.Adam()
    optimizer = optimizer.get_tensor()

    trainer = define_trainer(target, optimizer)

    pretrained_model = get_checkpoint(CHECKPOINT_PATH)
    trainer.restore(pretrained_model)

    # calculate mu and logvar for trainX and testX
    evaluator = define_evaluator(trainer, mu, logvar)
    train_mu_logvar = evaluator.predict({input_x: trainX})
    test_mu_logvar = evaluator.predict({input_x: testX})

    male_mu = 0
    female_mu = 0
    male_var = 0
    female_var = 0
    male_mu2 = 0
    female_mu2 = 0
    male_var2 = 0
    female_var2 = 0
    male_num = 0
    female_num = 0
    for i, y in enumerate(trainY):
        mu = train_mu_logvar[i][0]
        var = np.exp(train_mu_logvar[i][1])
        if y[0] == 0:
            male_mu += mu[0]
            male_var += var[0]
            male_mu2 += mu[1]
            male_var2 += var[1]
            male_num += 1
        else:
            female_mu += mu[0]
            female_var += var[0]
            female_mu2 += mu[1]
            female_var2 += var[1]
            female_num += 1

    male_mu /= (male_num)
    male_var /= (male_num)
    male_mu2 /= (male_num)
    male_var2 /= (male_num)
    female_mu /= (female_num)
    female_var /= (female_num)
    female_mu2 /= (female_num)
    female_var2 /= (female_num)

    print("MALE")
    print(" - mu1", male_mu*8.54962+164.923)
    print(" - var1", male_var*8.54962*8.54962)
    print(" - mu2", male_mu2*8.54962+164.923)
    print(" - var2", male_var2*8.54962*8.54962)
    print("FEMALE")
    print(" - mu1", female_mu*8.54962+164.923)
    print(" - var1", female_var*8.54962*8.54962)
    print(" - mu2", female_mu2*8.54962+164.923)
    print(" - var2", female_var2*8.54962*8.54962)

    return 0

if __name__ == '__main__':
    sys.exit(main())
