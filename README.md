# GAUSS
## Neural decision ensemble [GRU + DNN -> Extremely Randomized Trees] for predicting tabular data
### Usage:
```python
from gauss import *

gru = GRU(X=X_train)
nn = NN(X=X_train)
meta = Meta()
meta.fit(X_train,  y_train, gru, nn)
```
