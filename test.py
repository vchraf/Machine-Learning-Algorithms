from DL.DNN import NN

from sklearn import datasets
from sklearn.model_selection import train_test_split
data = datasets.load_digits()
X = data.data
y = data.target
# y = oneHot(y.astype("int"))
# optimizer = Adam()
# n_hidden = 100
# n_samples, n_features = X.shape
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=772)