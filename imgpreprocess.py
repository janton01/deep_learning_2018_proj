import numpy as np


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    x = x.reshape(x.shape[0], -1)
    sumx = x.sum(axis=1) / x.shape[1]
    x = x - sumx.reshape(-1, 1)
    return x


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    var = (x ** 2).sum(axis=1) / x.shape[1]
    var = var.reshape(-1, 1)
    x = scale * x / np.sqrt(var + bias)
    return x


def feature_zero_mean(x):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    meanx = x.sum(axis=0) / x.shape[0]
    x = x - meanx
    return x


def zca(x, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    U, S, V = np.linalg.svd((x.T).dot(x) / x.shape[0] + bias * np.eye(x.shape[1]))
    pca = (U.dot(np.diag(1. / np.sqrt(S)))).dot(U.T)
    zca_x = x.dot(pca)
    return zca_x


def cifar_10_preprocess(x, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    x = sample_zero_mean(x)
    x = gcn(x)
    x = feature_zero_mean(x)
    x = zca(x)
    x = x.reshape(x.shape[0], image_size, image_size)
    return x
