# This is a sample Python script.
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

#list(iris.keys())
#print(iris.DESCR)

X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(int)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state= 42)
log_reg.fit(X,y)

X_new = np.linspace(0,3, 1000).reshape(-1,1)
y_prob = log_reg.predict_proba(X_new)

plt.plot(X_new, y_prob[:,1], "g-", linewidth=2, label="Iris Virginica")
plt.plot(X_new, y_prob[:,0], "b--", linewidth=2, label="Not Iris Virginica")
plt.show()
