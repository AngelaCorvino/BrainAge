# pylint: disable=invalid-name, redefined-outer-name

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# Custom activation function
from features import Preprocessing


def step_wise(x, N = 4, a = 100):
    """Custom step-wise function to use as activation for RNN.

    Parameters
    ----------
    x : type
        Description of parameter `x`.
    N : integer
        Number of steps. Default is  4.
    a : integer
        Tuning parameter. Default is 100.
    Returns
    -------
    y : type
        Return function.

    """
    y = 1 / 2
    for j in range(1, N):
        y += (1 / (2 * (N - 1))) * (K.tanh(a * (x - (j / N))))
    return y


class Outliers:
    """Class identifying outliers.

    Parameters
    ----------

    """
    def __init__(self, X_train, X_test):
        """
        Constructur
        """
        self.X_train=X_train
        self.X_test=X_test
        self.model = self.make_autoencoder()


    def make_autoencoder(self):
        """Autoencoder trained comparing the output vector with the input features,
        using the Mean Squared Error (MSE) as loss function..

        Parameters
        ----------

        Returns
        -------
        model : type
            The trained model.

        history : type
            summary of how the model trained (training error, validation error).

        """
        get_custom_objects().update({"step_wise": Activation(step_wise)})

        inputs = Input(shape=self.X_train.shape[1])
        hidden = Dense(30, activation="tanh")(inputs)
        hidden = Dense(2, activation="step_wise")(hidden)
        hidden = Dense(30, activation="tanh")(hidden)
        outputs = Dense(self.X_train.shape[1], activation="linear")(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["MAE"])
        model.summary()
        return model

    def fit_autoencoder(self):

        history = self.model.fit(
            self.X_train, self.X_train, validation_split=0.4, epochs=100, batch_size=50, verbose=1
        )
        return history

    def outliers(self):
        """
        Identifies ouliers using autoencoder.
        """
        x_train_pred = model.predict(self.X_train)
        x_test_pred = model.predict(self.X_test)
        train_mae_loss = np.mean(
            np.abs(x_train_pred - np.array(self.X_train)), axis=1
        ).reshape((-1))
        test_mae_loss = np.mean(np.abs(x_test_pred - np.array(self.X_test)), axis=1).reshape(
            (-1)
        )
        anomalies = (test_mae_loss >= 0.5 * np.max(train_mae_loss)).tolist()
        histogram = train_mae_loss.flatten()
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Indices of anomaly samples: ", np.where(anomalies))
        print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))
        x_train_pred = model.predict(self.X_train)

        histogram = train_mae_loss.flatten()

        plt.hist(histogram, label="MAE Loss")

        plt.title("Mean Absolute Error Loss")

        plt.xlabel("Training MAE Loss (%)")
        plt.ylabel("Number of Samples")
        plt.show()


        # histogram1 = test_mae_loss.flatten()
        # plt.hist(histogram1,
        #                               label = 'MAE Loss')

        # plt.title('Mean Absolute Error Loss')
        # plt.xlabel("Test MAE Loss (%)")
        # plt.ylabel("Number of Samples")

        print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))

        return plt.show()


if __name__ == "__main__":

    def split(dataframe):
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
        return df_AS, df_TD

    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df = prep(df, "neuro", plot_option=False)
    df_AS, df_TD = split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df_TD, df_TD["AGE_AT_SCAN"], test_size=0.3, random_state=14
    )
    out = Outliers(X_train,X_test)
    out.fit_autoencoder()
    out.outliers()
