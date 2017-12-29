#### Description
Currently `ASPRAwL` is just a library for doing some more complex feature selection automation in PySpark.  

The environment I use this on is a cluster with PySpark 1.6.0 

##### Feature Selection

* RandomForest coverage selector.  This selects the top percent of features. 
```pythonstub
from asprawl.feature_selection.coverage import RandomForestCoverageSelector
va = VectorAssembler(inputCols=cols, outputCol='features')
si = StringIndexer(inputCol="y", outputCol="label")
t_pipe = Pipeline(stages=[va, si])
t_model = t_pipe.fit(df)
t_data = t_model.transform(df).select(["features", "label"])

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
est = RandomForestCoverageSelector(estimator=rf, pct=0.95)
selector = est.fit(t_data)
s_data = selector.transform(t_data).select(["sliced", "label"])
s_data.show()
indices = [i for i in selector.getIndices()]
log.warn("selected features:")
log.warn(",".join(np.array(cols)[indices]))
```
* RandomForestCartSelector, which under the hood selects features based on CART model of cross-validated variable
importances.  
```pythonstub
from asprawl.feature_selection.cart_selection import RandomForestCartSelector
va = VectorAssembler(inputCols=cols, outputCol='features')
si = StringIndexer(inputCol="y", outputCol="label")
t_pipe = Pipeline(stages=[va, si])
t_model = t_pipe.fit(df)
t_data = t_model.transform(df).select(["features", "label"])

rf = RandomForestClassifier(labelCol="label", featuresCol="features")
est = RandomForestCartSelector(estimator=rf, inputCol='features', outputCol='sliced', n_folds=3, seed=3233)
selector = est.fit(t_data)
s_data = selector.transform(t_data).select(["sliced", "label"])
s_data.show()
indices = [i for i in selector.getIndices()]
log.warn("selected features:")
log.warn(",".join(np.array(cols)[indices]))
```
* ForwardSelector which selects taking a forward path through feature importances
```pythonstub
from asprawl.feature_selection.forward import ForwardSelector
```