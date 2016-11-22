from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

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


def main():
    gnb = GaussianNB()
    gnb.fit(trainX, trainY)

    print(gnb.theta_)
    print(gnb.sigma_)

if __name__ == '__main__':
    sys.exit(main())
