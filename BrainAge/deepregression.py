# pylint: disable=invalid-name, redefined-outer-name, import-error
"""
Module implements a MLP and fits it on given dataframe.
"""
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.models import Model
from sklearn.base import BaseEstimator


class DeepRegression(BaseEstimator):
    """
    Class describing deep regression model.

    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    
    Parameters
    ----------
    plot_loss : bool, default=False
        When set to ``True``, plots the training and validation loss curves of the trained model

    Attributes
    ----------
    model : object
    """

    def __init__(
        self,
        epochs=100,
        drop_rate=0.2,
        plot_loss=False,
    ):
        self.epochs = epochs
        self.drop_rate = drop_rate
        self.plot_loss = plot_loss
        super().__init__()

    def fit(self, X, y):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.


        Returns
        -------
        self : object
            Fitted Estimator.
        """
        inputs = Input(shape=X.shape[1])
        hidden = Dense(128, activation="relu")(inputs)
        hidden = Dense(64, activation="relu")(hidden)
        hidden = Dense(12, activation="relu")(hidden)
        hidden = Dropout(self.drop_rate)(hidden)
        hidden = Dense(12, activation="relu")(hidden)
        outputs = Dense(1, activation="linear")(hidden)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss="mean_absolute_error", optimizer="adam", metrics=["MAE"]
        )
        self.model.summary()
        # Define callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=1
        )
        history = self.model.fit(
            X,
            y,
            validation_split=0.3,
            epochs=self.epochs,
            callbacks=[early_stopping],
            batch_size=32,
            verbose=0,
        )

        if self.plot_loss is True:

            # This parameter allows to plot the training and validation loss
            # curves of the trained model, enabling visual diagnosis of
            # underfitting (bias) or overfitting (variance).

            training_validation_loss = pd.DataFrame.from_dict(
                history.history, orient="columns"
            )

            plt.figure(figsize=(8, 8))
            plt.scatter(
                training_validation_loss.index,
                training_validation_loss["loss"],
                marker=".",
                label="Training Loss",
            )
            plt.scatter(
                training_validation_loss.index,
                training_validation_loss["val_loss"],
                marker=".",
                label="Validation Loss",
            )

            plt.title("Training and Validation Loss")
            plt.xlabel("Epoch",fontsize=20)
            plt.ylabel("Loss",fontsize=20)
            plt.tick_params(axis="x", which="major", labelsize=18)
            plt.tick_params(axis="y", which="major", labelsize=18)
            plt.legend(loc="upper right", fontsize=14)    
            plt.legend(title='{}'.format(super().get_params(self.model)),loc="upper right", fontsize=14)
            plt.show()

    def predict(self, X):
        """implement predict method

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to dtype=np.float32.

        Returns
        -------
        self.model.predict :ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes..

        """
        return self.model.predict(X)
