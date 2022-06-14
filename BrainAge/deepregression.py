from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model

from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import pandas as pd


from features import Preprocessing


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
    model



    """

    def __init__(
        self,
        epochs=100,
        plot_loss=False,
    ):
        self.epochs = epochs
        self.plot_loss = plot_loss  # io toglierei anche questi attributi, passandoli direttamente alle funzioni che li usano

    def fit(self, X, y):
        """
        Fit linear model.
        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
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
        hidden = Dense(12, activation="relu")(hidden)
        hidden = Dense(12, activation="relu")(hidden)
        hidden = Dense(12, activation="relu")(hidden)
        outputs = Dense(1, activation="linear")(hidden)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss="mean_absolute_error", optimizer="adam", metrics=["MAE"]
        )
        self.model.summary()
        history = self.model.fit(
            X, y, validation_split=0.3, epochs=self.epochs, batch_size=50, verbose=0
        )
        return self

        if self.plot_loss == True:
            """
            This parameter allows to plot the training and validation loss
            curves of the trained model, enabling visual diagnosis of
            underfitting (bias) or overfitting (variance).


            Returns
            -------
            fig: a visual representation of the model's training loss and
            validation loss curves.
            """
            print("problem here")
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
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

    def predict(self, X):
        return self.model.predict(X)

        #     def reconstruction_error(self,model):
        # """
        # This function calculates the reconstruction error and displays a histogram of
        # the training mean absolute error.
        # Arguments:
        # model: the trained  model
        #   x_train: 3D data to be used in model training (dataframe).
        #   Returns:
        #   fig: a visual representation of the training MAE distribution.
        # """
        #
        # x_train_pred = model.predict(self.X_train)
        # train_mae_loss = np.mean(np.abs(x_train_pred - np.array(self.X_train)), axis = 1)
        # histogram = train_mae_loss.flatten()
        # plt.hist(histogram,
        #                               label = 'MAE Loss')
        #
        # plt.title('Mean Absolute Error Loss')
        # plt.xlabel("Training MAE Loss (%)")
        # plt.ylabel("Number of Samples")
        #
        #
        # print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))
        #
        # return plt.show()


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    dataframe = prep(df, "raw", False)
    a = DeepRegression(epochs=10)
    a.fit(dataframe.drop(["AGE_AT_SCAN"], axis=1), dataframe["AGE_AT_SCAN"])
