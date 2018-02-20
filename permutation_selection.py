from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.sparse import random as sparse_rand
import scipy.stats as st
import pandas as pd


class BasePermutation(BaseEstimator, TransformerMixin):
    def __init__(self, model, test_size, random_state, score_function):
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.score_function = score_function

    def fit(self, X, y):
        X = self.modify_data(X)
        train, test = self.set_cv(X)
        self.model.fit(X[train], y[train])
        self.permutation_score(X[test], y[test])
        return self

    def transform(self, X):
        return X

    def split_data(self, **kwargs):
        cv = ShuffleSplit(**kwargs)
        train, test = next(iter(cv))
        return train, test

    def score(self, X, y):
        yhat = self.model.predict(X)
        return self.score_function(y, yhat)

    def set_cv(self, X):
        return self.split_data(n=X.shape[0], n_iter=1, test_size=self.test_size, random_state=self.random_state)

    def modify_data(self, X):
        return X

    def permutation_score(self, X, y):
        pass


class SparkPermutationSelection(BasePermutation):
    def __init__(self, model, test_size, random_state, score_function, spark_context, iter=10, sentinel_density=0.8):
        super(SparkPermutationSelection, self).__init__(model, test_size, random_state, score_function)
        self.spark_context = spark_context
        self.iter = iter
        self.sentinel_density = sentinel_density

    def modify_data(self, X):
        sentinel = sparse_rand(X.shape[0], 1, density=self.sentinel_density,
                               format='csr', random_state=self.random_state)
        X = hstack([X, sentinel], format='csr')
        return X

    def permutation_score(self, X, y):
        column_indx = np.arange(X.shape[1])
        columns = self.spark_context.parallelize(column_indx, len(column_indx))
        data_bc = self.spark_context.broadcast(dict(data=X.data, indptr=X.indptr, indices=X.indices, shape=X.shape, y=y))
        model_bc = self.spark_context.broadcast(self.model)
        scoring_func_bc = self.spark_context.broadcast(self.score_function)
        pred = self.model.predict(X)
        base_score = self.score_function(y, pred)
        print(base_score)
        base_score_bc = self.spark_context.broadcast(base_score)

        def score(c_i):
            X_test = csr_matrix((data_bc.value['data'], data_bc.value['indices'], data_bc.value['indptr']),
                                shape=data_bc.value['shape'])
            X_test = X_test.copy()
            y_test = data_bc.value['y']
            model = model_bc.value
            tmp_scores = np.zeros(10)
            for i in range(10):
                idx = np.random.permutation(y_test.shape[0])
                arr = X_test[:, c_i].todense()
                X_test[:, c_i] = arr[idx]
                pred = model.predict(X_test)
                tmp_scores[i] = -(base_score_bc.value - scoring_func_bc.value(y_test, pred))
            lcl, ucl = st.t.interval(0.99, len(tmp_scores)-1, loc=np.mean(tmp_scores), scale=st.sem(tmp_scores))
            return c_i, np.mean(tmp_scores), lcl, ucl

        scores = columns.map(score).collect()
        ci, mu, lcl, ucl = zip(*scores)

        df = pd.DataFrame({'indx': ci, 'mu': mu, 'lcl': lcl, 'ucl': ucl})
        df.sort_values('indx', inplace=True)
        self.scores = df.mu.values
        self.lcl = df.lcl.values
        self.ucl = df.ucl.values

        sentinel_score = self.ucl[-1]
        indx_gt_sentinel = np.where((self.lcl > sentinel_score) & (self.lcl > 0))[0]
        self.support = np.zeros(X.shape[1]-1, dtype=np.bool)
        self.support[indx_gt_sentinel] = True

    def transform(self, X):
        return X[:, self.support]


def test():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from pyspark.context import SparkContext
    spark_context = SparkContext()
    X, y, coef = make_regression(n_samples=10000, n_features=20, n_informative=6, noise=2., random_state=25, coef=True)
    X = csr_matrix(X)

    model = SparkPermutationSelection(model=Ridge(), test_size=0.2, random_state=325,
                                      score_function=mean_absolute_error, spark_context=spark_context)
    model.fit(X, y)

if __name__ == '__main__':
    test()
