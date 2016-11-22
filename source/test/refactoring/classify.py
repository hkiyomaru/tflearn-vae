import tensorflow as tf
import tflearn

from VAE import VAE
from SVC import SupportVectorClassifier

from dataset import Dataset, Datasets

import pickle
import sys


# loading data
iris = pickle.load(open('iris.pkl', 'rb'))
trainX, trainY, testX, testY = iris.load_data()

# network parameters
input_dim = 4 # height data input
latent_dim = 4

# define training parameters
batch_size = 50


# flow of SVM classification
def main():
    global trainX, trainY, testX, testY

    vae = VAE(input_dim, latent_dim)

    input_x = tflearn.input_data(shape=(None, input_dim), name='input_x')
    optimizer = tflearn.optimizers.Adam().get_tensor()

    trainer = vae.return_trainer(input_x, optimizer, batch_size)
    trainer.restore(vae.get_checkpoint())

    # calculate mu and logvar for trainX and testX
    evaluator = vae.return_evaluator(trainer)
    train_mu_logvar = evaluator.predict({input_x: trainX})
    test_mu_logvar = evaluator.predict({input_x: testX})

    # classification
    classifier = SupportVectorClassifier()
    classifier.fit(train_mu_logvar, trainY)

    # evaluate
    classifier.score(test_mu_logvar, testY)


# entry point
if __name__ == '__main__':
    sys.exit(main())
