import mglearn.datasets
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split # 自动将数据集分为训练集与数据集
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
from mglearn import cm3
import matplotlib.pyplot as plt


def iris_knn():
    # knn判断iris数据集花的类别
    iris_dataset = load_iris()
    print(iris_dataset.keys())
    # print(iris_dataset['DESCR'])

    data = iris_dataset['data']
    target = iris_dataset['target']
    # name = iris_dataset['target_names']
    # length = len(data)
    # print('数据   类别')
    # for i in range(length):
    #     print("{}    {}".format(data[i], target[i]))

    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

    # iris_dataset.feature_names = iris_dataset['feature_names']
    # iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    # grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(10, 6), marker='o', hist_kwds={'bins': 20}, s=60,
    #                                  alpha=0.8, cmap=cm3)
    # plt.show()

    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print("Test set predictions: \n{}".format(y_pred))
    print("Test set score: {:.3f}".format(np.mean(y_pred == y_test)))
    # print("Test set score: {:.3f}".format(knn.score(X_test, y_test)))


def forge_knn():
    # forge数据集画出k近邻分类决策边界
    X, y = mglearn.datasets.make_forge()
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        # n_neighbors为索引为 1, 3, 9
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title('{} neighbor(s)'.format(n_neighbors))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend(loc=3)
    plt.show()


def boston_linear_regression():
    # boston数据集进行 线性回归, 岭回归, lasso
    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    lr = LinearRegression().fit(X_train, y_train)
    # print("Linear regression coefficent: {}, intercept: {}".format(lr.coef_, lr.intercept_))
    print("Linear regression training set score: {:.3f}".format(lr.score(X_train, y_train)))
    print("Linear regression test set score: {:.3f}".format(lr.score(X_test, y_test)))
    ridge = Ridge(alpha=0.1).fit(X_train, y_train)
    # print("Ridge coefficent: {}, intercept: {}".format(ridge.coef_, ridge.intercept_))
    print("Ridge training set score: {:.3f}".format(ridge.score(X_train, y_train)))
    print("Ridge test set score: {:.3f}".format(ridge.score(X_test, y_test)))
    lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    # print("Lasso coefficent: {}, intercept: {}".format(lasso.coef_, lasso.intercept_))
    print("Lasso training set score: {:.3f}".format(lasso.score(X_train, y_train)))
    print("Lasso test set score: {:.3f}".format(lasso.score(X_test, y_test)))


def forge_linear_classification():
    X, y = mglearn.datasets.make_forge()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    for model, ax in zip([LinearSVC(C=0.01, penalty='l2'), LogisticRegression(C=0.01, penalty='l2')], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{}".format(clf.__class__.__name__))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend()
    plt.show()


def a():
    ax = plt.subplot()
    mglearn.plots.plot_animal_tree(ax=ax)
    plt.show()


if __name__ == '__main__':
    # iris_knn()
    # forge_knn()
    # boston_linear_regression()
    # forge_linear_classification()
    a()
