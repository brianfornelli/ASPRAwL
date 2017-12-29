from pyspark.ml.pipeline import Estimator
from pyspark.sql.functions import rand
import numpy as np


class ForwardSelector(Estimator):
    def __init__(self, estimator, evaluator, selector, early_stopping=3):
        """ForwardSelector performs forward selection along grid-path of selected features, adding when there is lift

        Parameters
        ---------------------
        estimator:
            PySpark estimator
        evaluator:
            PySpark evaluation object
        selector:
            Fit Asprawl selector object
        early_stopping: int, default 3
            how many features to be considered with zero lift before exiting

        """
        super(ForwardSelector, self).__init__()
        self.estimator = estimator
        self.evaluator = evaluator
        self.early_stopping = early_stopping
        self.n_folds = selector.n_folds
        self.seed = selector._seed
        self.sel = selector

    def _fit(self, dataset):
        mu = self.sel.mu
        MASTER_FEATURES = np.arange(mu.shape[0])
        MASTER_FEATURES = MASTER_FEATURES[self.sel.indices]
        mu = mu[MASTER_FEATURES]
        sorted_indx = np.argsort(mu)[::-1]
        good_features = []
        mx_score = 0.0
        for i, ftr_indx in enumerate(sorted_indx):
            good_features.append(MASTER_FEATURES[ftr_indx])
            cv = CrossValidationPerformance(estimator=self.estimator, evaluator=self.evaluator, n_folds=3)
            cv_model = cv.fit(dataset)
            score = np.mean(cv_model.scores)
            if score > mx_score:
                ctr = 0
                mx_score = score
            else:
                good_features.pop()
                ctr += 1
            if ctr > self.early_stopping:
                break
        self.good_features = good_features
        return self


class CrossValidationPerformance(Estimator):
    def __init__(self, estimator, evaluator, n_folds=3, seed=3233):
        """CrossValidationPerformance performs cross-validation and returns the evaluated performance

        Parameters
        ---------------------
        estimator:
            PySpark estimator
        evaluator: pyspark.ml.evaluation
            PySpark evaluation object
        n_folds: int, default 3
            number of cross-validation folds
        seed: int
            random seed for cross-validation

        """
        super(CrossValidationPerformance, self).__init__()
        self.estimator = estimator
        self.n_folds = n_folds
        self._seed = seed
        self.evaluator = evaluator
        self._rnd = self.uid + "_rand"
        self.scores = list()

    def _fit(self, dataset):
        df = dataset.select("*", rand(self._seed).alias(self._rnd))
        for i, test_instances in enumerate(self._cv(df)):
            train = df.filter(~test_instances).drop(self._rnd)
            model = self.estimator.fit(train)
            test = df.filter(test_instances).drop(self._rnd)
            scored = model.transform(test)
            self.scores.append(self.evaluator.evaluate(scored))
        return self

    def _cv(self, dataset):
        h = 1.0 / self.n_folds
        for fold in range(self.n_folds):
            lb, ub = fold * h, (fold + 1) * h
            test_filter = (dataset[self._rnd] > lb) & (dataset[self._rnd] < ub)
            yield test_filter
