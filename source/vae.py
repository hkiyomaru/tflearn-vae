import tensorflow as tf
import tflearn

from dataset import Dataset, Datasets

import pickle


# loading data
height = pickle.load(open('height.pkl', 'rb'))
trainX, trainY, testX, testY = height.load_data()

# network parameters
input_dim = 1 # height data input
encoder_hidden_dim = 16
decoder_hidden_dim = 16
latent_dim = 2

# paths
TENSORBOARD_DIR='experiment/'
CHECKPOINT_PATH='out_models/'

# training parameters
n_epoch = 10
batch_size = 50

# encoder
input_x = tflearn.input_data(shape=(None, input_dim), name='input_x')
encoder = tflearn.fully_connected(input_x, encoder_hidden_dim, activation='relu', regularizer='L2')
mu_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear', regularizer='L2')
logvar_encoder = tflearn.fully_connected(encoder, latent_dim, activation='linear', regularizer='L2')

# sampling
epsilon = tf.random_normal(tf.shape(logvar_encoder), dtype=tf.float32, name='epsilon')
std_encoder = tf.exp(tf.mul(0.5, logvar_encoder))
z = tf.add(mu_encoder, tf.mul(std_encoder, epsilon))

# decoder
decoder = tflearn.fully_connected(z, decoder_hidden_dim, activation='relu', regularizer='L2')
x_hat = tflearn.fully_connected(decoder, input_dim, activation='linear', regularizer='L2')

# calculating loss
kl_divergence = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.square(mu_encoder) - tf.exp(logvar_encoder), reduction_indices=1)
bce = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, input_x), reduction_indices=1)
loss = tf.reduce_mean(tf.add(kl_divergence, bce))

# optimization
optimizer = tflearn.Adam()
step = tflearn.variable("step", initializer='zeros', shape=[])
optimizer.build(step_tensor=step)
optim_tensor = optimizer.get_tensor()

# trainer
trainop = tflearn.TrainOp(loss=loss,
                          optimizer=optim_tensor,
                          batch_size=batch_size,
                          metric=None,
                          name='vae_trainer')

trainer = tflearn.Trainer(train_ops=trainop,
                          tensorboard_dir=TENSORBOARD_DIR,
                          tensorboard_verbose=2,
                          checkpoint_path=CHECKPOINT_PATH,
                          max_checkpoints=1)

# evaluator
evaluator = tflearn.Evaluator([x_hat], session=trainer.session)

# launch the graph
with tf.Graph().as_default():
    trainer.fit(feed_dicts={input_x: trainX},
                val_feed_dicts={input_x: testX},
                n_epoch=n_epoch,
                show_metric=False,
                snapshot_epoch=True,
                shuffle_all=True,
                run_id='VAE')

    print(testX[:10])
    print(evaluator.predict({input_x: testX})[:10])
