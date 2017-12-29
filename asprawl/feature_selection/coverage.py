from pyspark.ml.pipeline import Estimator
from pyspark.ml.feature import VectorSlicer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
import numpy as np
import pandas as pd


class RandomForestCoverageSelector(Estimator, HasInputCol, HasOutputCol):
    """RandomForestCoverageSelector - selection based on percent of feature importances

    Parameters
    ---------------------
    estimator:
        pyspark.ml.RandomForest[Classifier|Regression]
    inputCol: string, default ="features"
        input features column
    outputCol: string, default "sliced"
        output subset column
    pct: float, default 0.95
        pct threshold
    """
    def __init__(self, estimator, inputCol='features', outputCol='sliced', pct=0.95):
        super(RandomForestCoverageSelector, self).__init__()
        self.estimator = estimator
        self.pct = pct
        self.inputCol = inputCol
        self.outputCol = outputCol

    def _fit(self, dataset):
        model = self.estimator.fit(dataset)
        feature_importances = model._call_java("featureImportances")
        topfeatures = self._get_top_features(feature_importances)
        indices = topfeatures.loc[topfeatures['cumulative_feature_importances'] < self.pct, 'indices'].values
        indices.sort()
        self.indices = indices
        vs = VectorSlicer(inputCol=self.inputCol, outputCol=self.outputCol, indices=[i.item() for i in self.indices])
        return vs

    def _get_top_features(self, fi):
        '''rank the cumulative feature importance'''
        values = fi.values/np.sum(fi.values)
        df = pd.DataFrame({'indices': fi.indices, 'feature_importances': values})
        df.insert(0, 'rank', df['feature_importances'].rank(method='min', ascending=False).astype(int))
        df.sort_values('rank', inplace=True)
        df['cumulative_feature_importances'] = df['feature_importances'].cumsum()
        return df
