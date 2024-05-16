# GAUSS
## Neural decision ensemble [GRU + DNN -> Extremely Randomized Trees] for predicting tabular data
### Usage:
```python
from gauss import *

gru = GRU(X=X_train)
nn = NN(X=X_train)
meta = Meta()
meta.fit(X_train,  y_train, gru, nn)
# -- #

# Convert X_test to NumPy array and generate predictions from GRU and NN models
gru_pred = gru.model.predict(X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)))
nn_pred = nn.model.predict(X_test.values)

# Concatenate predictions and pass to Meta model for prediction
meta_input = np.concatenate([gru_pred, nn_pred], axis=1)
y_pred = meta.predict(meta_input)
```
