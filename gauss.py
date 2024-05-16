# gauss.py
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class BaseModel:
    def __init__(self, input_shape: tuple):
        """
        Initialize the base model.

        Args:
            input_shape (tuple): The shape of the input data.
        """
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.input_layer = tf.keras.layers.Input(shape=input_shape)
        self.normalized_input = self.normalizer(self.input_layer)
        self.model: tf.keras.Model = None

    def compile(self, loss: str = 'mean_absolute_error', optimizer: str = 'nadam') -> None:
        """
        Compile the model.

        Args:
            loss (str, optional): The loss function. Defaults to 'mean_absolute_error'.
            optimizer (str, optional): The optimizer. Defaults to 'nadam'.
        """
        self.model.compile(loss=loss, optimizer=optimizer)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, validation_split: float = 0.2, callbacks: list = None) -> None:
        """
        Fit the model.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.
            epochs (int, optional): The number of epochs. Defaults to 100.
            batch_size (int, optional): The batch size. Defaults to 32.
            validation_split (float, optional): The validation split. Defaults to 0.2.
            callbacks (list, optional): The callbacks. Defaults to None.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
        self.normalizer.adapt(X)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping])

class GRU(BaseModel):
    def __init__(self, X: np.ndarray):
        """
        Initialize the GRU model.

        Args:
            X (np.ndarray): The input data.
        """
        if not isinstance(X, np.ndarray):
            X = X.values
        if len(X.shape) == 2:
            input_shape = (X.shape[1], 1)  # IF 2D (X.shape[1], 1)
        else:
            input_shape = (X.shape[1], X.shape[2])  # IF 3D
        super().__init__(input_shape=input_shape)
        print(f"-> [GRU] Input shape is: {input_shape}")
        print(f"-> [GRU] Initilizing with topology: 50/tanh|50/tanh|1")
        self.gru_layer1 = tf.keras.layers.GRU(units=50, return_sequences=True, activation='tanh')(self.normalized_input)
        self.gru_layer2 = tf.keras.layers.GRU(units=50, activation='tanh')(self.gru_layer1)
        self.output_layer = tf.keras.layers.Dense(units=1)(self.gru_layer2)
        self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.output_layer)
        
class NN(BaseModel):
    def __init__(self, X: np.ndarray):
        """
        Initialize the NN model.

        Args:
            X (np.ndarray): The input shape.
        """
        if not isinstance(X, np.ndarray):
            X = X.values
        super().__init__(input_shape=X.shape[1:])
        print(f"-> [NN] Input shape is: {X.shape[1:]}")
        print(f"-> [NN] Initilizing with topology: 128/relu|64/relu|1/linear")
        self.nn_layer1 = tf.keras.layers.Dense(units=128, activation='relu')(self.normalized_input)
        self.nn_layer2 = tf.keras.layers.Dense(units=64, activation='relu')(self.nn_layer1)
        self.output_layer = tf.keras.layers.Dense(units=1, activation='linear')(self.nn_layer2)
        self.model = tf.keras.models.Model(inputs=self.input_layer, outputs=self.output_layer)

class Meta:
    def __init__(self):
        """
        Initialize the Meta model.
        """
        self.meta_model: ExtraTreesRegressor = None

    def fit(self, X: np.ndarray, y: np.ndarray, gru_model: GRU, nn_model: NN) -> None:
        """
        Fit the Meta model.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The target data.
            gru_model (GRU): The GRU model.
            nn_model (NN): The NN model.
        """
        # Compile the models
        gru_model.compile(loss='mean_absolute_error', optimizer='nadam')
        nn_model.compile(loss='mean_absolute_error', optimizer='nadam')

        if not isinstance(X, np.ndarray):
            X = X.values
        X_3d = X.reshape((X.shape[0], X.shape[1], 1))

        # Train the GRU and NN models
        gru_model.fit(X_3d, y)
        nn_model.fit(X, y)

        # Generate predictions from the GRU and NN models
        gru_pred = gru_model.model.predict(X_3d)
        nn_pred = nn_model.model.predict(X)

        # Concatenate the predictions and targets into a new matrix
        new_data = np.column_stack((gru_pred.flatten(), nn_pred.flatten(), y))

        # Fit the Meta model using the new data
        self.meta_model = ExtraTreesRegressor(n_estimators=20, criterion='absolute_error')
        self.meta_model.fit(new_data[:, :2], new_data[:, 2])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Meta model.

        Args:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target values.
        """
        return self.meta_model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, gru_model: GRU, nn_model: NN) -> dict:
        """
        Evaluate the models.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series): The target data.
            gru_model (GRU): The GRU model.
            nn_model (NN): The NN model.

        Returns:
            dict: The evaluation metrics.
        """
        # Compile the models
        gru_model.compile(loss='mean_absolute_error', optimizer='nadam')
        nn_model.compile(loss='mean_absolute_error', optimizer='nadam')

        X_3d = X.values.reshape((X.shape[0], 1, X.shape[1]))

        # Train the GRU and NN models
        gru_model.fit(X_3d, y)
        nn_model.fit(X, y)

        # Generate predictions from the GRU and NN models
        gru_pred = gru_model.model.predict(X_3d)
        nn_pred = nn_model.model.predict(X)

        # Concatenate the predictions to generate the final prediction
        meta_input = np.concatenate([gru_pred, nn_pred], axis=1)
        y_pred = self.meta_model.predict(meta_input)

        # Calculate evaluation metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)

        return {
            'MSE': mse,
            'MAE': mae,
            'MAE%': mape
        }
    
    def save(self):
        # Save the Meta model
        joblib.dump(self, 'meta_model.pkl')

    @staticmethod
    def evaluate_data(X: pd.DataFrame, y: pd.Series) -> None:
        """
        Evaluate the models.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.Series): The target data.
        """
        gru_model = GRU(X)
        nn_model = NN(X.shape)
        meta_model = Meta()
        meta_model.fit(X, y, gru_model, nn_model)
        # Make predictions using the Meta model
        y_pred = meta_model.predict(X)
        # Evaluate the performance of the Meta model
        mae = mean_absolute_error(y, y_pred)
        print(f'Mean Absolute Error: {mae}')
