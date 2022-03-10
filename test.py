import numpy as np

from DL.DNN.NN import NN
from DL.DNN.Optimisation.Adam import Adam
from DL.DNN.Layer.Dense import Dense
from DL.DNN.Layer.Activation import Activation 
from DL.DNN.Loss.SquareLoss import SquareLoss

from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils.utils import oneHot

data = datasets.load_digits()
X = data.data
y = data.target
y = oneHot(y.astype("int"))
optimizer = Adam()
n_hidden = 100
n_samples, n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=72)

mlp = NN(optimizer= optimizer,loss= SquareLoss,validationData=(X_test, y_test))
mlp.add(Dense(n_hidden, inputShape=(n_features, )))
mlp.add(Activation('relu'))
mlp.add(Dense(10))
mlp.add(Activation('sigmoid'))
train_err, val_err = mlp.fit(X_train, y_train, n_epochs=50, batch_size=256)