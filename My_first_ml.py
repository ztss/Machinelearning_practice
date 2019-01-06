from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# load_iris() is a bunch object, similar to a dict
iris = load_iris()

# this function extracts 75% of the rows in the data as the training set,together with the corresponding
# labels for this data. The remaining 25% of the data,together with the remaining labels are declared as the test set
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

# inspect our data
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("iris_pairplot")


for i in range(3):
    for j in range(3):
        ax[i, j].scatter(X_train[:, j], X_train[:, i+1], c=y_train, s=60)
        ax[i, j].set_xticks(())
        ax[i, j].set_yticks(())
        if i == 2:
            ax[i, j].set_xlabel(iris['feature_names'][j])
        if j == 0:
            ax[i, j].set_ylabel(iris['feature_names'][i+1])
        if j > i:
            ax[i, j].set_visible(False)
# show the fig
# plt.show()

# The knn object encapsulates the algorithm to build the model from the training tdata.
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# KNeighborsClassifier(algorithm='auto',leaf_size=30,metric='minkowski',metric_params=None,n_jobs=1,n_neighbors=1,p=2,
#                     weights='uniform')

# predictions using this model on new data.
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(prediction)

# we can measure how well the model works by compute the accuracy, which is the fraction of flowers for which the right
# species was predicted
y_pred = knn.predict(X_test)
print(np.mean(y_pred == y_test))

# can also use the score method of the knn object, which will compute the test set accuracy for us
accuracy = knn.score(X_test, y_test)
print(accuracy)
