from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

iris = load_iris()
X = iris.data
y = iris.target

# knn = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# print(scores.mean())
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    #loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    #k_scores.append(loss.mean())
    k_scores.append(scores.mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()