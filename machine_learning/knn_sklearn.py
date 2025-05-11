from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

if __name__ == "__main__":

    iris = load_iris()
    X, y, f_names = iris.data, iris.target, iris.feature_names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=10
    )

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print(knn_model.score(X_test, y_test))
    print(y_pred)
