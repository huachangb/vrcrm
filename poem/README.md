# POEM
Code in this module is written by A. Swaminathan and T. Joachims
Code can be found at https://www.cs.cornell.edu/~adith/POEM/ under ICML'15

## Changelog
```All files```
- Converted syntax for xrange and print to their Python3 equivalent.
- Changed numpy.int to numpy.int16

```DatasetReader.py```
- Changed train_test_split to correct function

```Skylines.py```
- Changed sklearn.grid_search to model_selection
- Added ```solver=liblinear``` to CRF, since logistic regression used liblinear as default in 2015.
- Changed order of ```if (bestPerformance > predictionError) or (bestPerformance is None):```



