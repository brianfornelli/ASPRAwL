from pyspark.ml.pipeline import Estimator
from pyspark.ml.feature import VectorSlicer
from pyspark.sql.functions import rand
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from collections import defaultdict


class RandomForestCartSelector(Estimator, HasInputCol, HasOutputCol):
    """RandomForestCartSelector performs feature selection using a CART estimator on the cross-validated feature
    importances.

    See Robin Genuer, Jean-Michel Poggi, Christine Tuleau-Malot. Variable selection using Random Forests.  Pattern
    Recognition Letters, Elsevier, 2010, 31 (14), pp.2225-2236. <hal-00755489>

    Parameters
    ---------------------
    estimator:
        pyspark.ml.RandomForest[Classifier|Regression]
    inputCol: string, default ="features"
        input features column
    outputCol: string, default "sliced"
        output subset column
    n_folds: int, default 3
        number of cross-validation folds
    seed: int
        random seed for cross-validation

    """
    def __init__(self, estimator, inputCol='features', outputCol='sliced', n_folds=3, seed=3233):
        super(RandomForestCartSelector, self).__init__()
        self.estimator = estimator
        self.n_folds = n_folds
        self._seed = seed
        self.inputCol = inputCol
        self.outputCol = outputCol
        self._rnd = self.uid + "_rand"

    def _fit(self, dataset):
        df = dataset.select("*", rand(self._seed).alias(self._rnd))
        feature_importances = list()
        for i, test_instances in enumerate(self._cv(df)):
            train = df.filter(~test_instances).drop(self._rnd)
            model = self.estimator.fit(train)
            topfeatures = RandomForestCartSelector._get_top_features(model)
            feature_importances.append(topfeatures)
        self.indices = self._set_threshold(feature_importances)
        vs = VectorSlicer(inputCol=self.inputCol, outputCol=self.outputCol, indices=[i.item() for i in self.indices])
        return vs

    def _cv(self, dataset):
        h = 1.0 / self.n_folds
        for fold in range(self.n_folds):
            lb, ub = fold * h, (fold + 1) * h
            test_filter = (dataset[self._rnd] > lb) & (dataset[self._rnd] < ub)
            yield test_filter

    @staticmethod
    def _get_top_features(model):
        fi = model._call_java("featureImportances")
        ftrs = model._call_java("numFeatures")
        iscores = defaultdict(float)
        iscores.update({i: v for i, v in zip(fi.indices, fi.values)})
        out = pd.DataFrame({"indx": np.arange(ftrs), "val": [iscores[k] for k in np.arange(ftrs)]})
        return out["val"].values

    def _set_threshold(self, vi):
        mu = np.mean(vi, 0)
        std = np.std(vi, 0)
        sorted_indx = np.argsort(mu)[::-1]
        sorted_std = std[sorted_indx]
        x = np.arange(sorted_std.shape[0])
        cart = DecisionTreeRegressor()
        #Fit CART to scale of vi
        cart.fit(x[:, np.newaxis], sorted_std)
        p = cart.predict(x[:, np.newaxis])
        #self.mu = [e.item() for e in mu]
        #self.std = [e.item() for e in std]
        self.mu = mu
        self.std = std
        self.p = p
        self.minimum_score = np.min(p)
        return [i for i in np.where(std > self.minimum_score)[0]]
