import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import classification_report


class SupportVectorClassifier(object):
    def __init__(self):
        self.estimator = SVC(C=1e6)

    def fit(self, train_mu_logvar, trainY):
        train_mu_logvar = self.reshaper(train_mu_logvar)
        self.estimator.fit(train_mu_logvar, trainY)

    def predict(self, test_mu_logvar):
        test_mu_logvar = self.reshaper(test_mu_logvar)
        predictions = self.estimator.predict(test_mu_logvar)
        return predictions

    def reshaper(self, mu_logvar):
        mu_logvar = np.asarray(mu_logvar).astype(np.float32)
        mu_logvar = np.concatenate((mu_logvar[:,0], mu_logvar[:,1]), axis=1)
        return mu_logvar

    def score(self, test_mu_logvar, testY):
        predictions = self.predict(test_mu_logvar)
        print(classification_report(predictions, testY))
