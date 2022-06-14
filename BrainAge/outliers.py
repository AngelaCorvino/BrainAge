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


def step_wise(x):
    """
    Custom step-wise function to use as activation for RNN
    """
    N = 4
    y = 1 / 2
    for j in range(1, N):
        y += (1 / (2 * (N - 1))) * (K.tanh(100 * (x - (j / N))))
    return y


class Outliers:
    """Class identifying outliers.

    Parameters
    ----------

    """

    def make_autoencoder(self, X_train):
        """Autoencoder trained comparing the output vector with the input features,
        using the Mean Squared Error (MSE) as loss function.

        Returns
        -------
        type
            Description of returned object.
        model : the trained model.
        history : a summary of how the model trained (training error, validation error).

        """
        get_custom_objects().update({"step_wise": Activation(step_wise)})

        inputs = Input(shape=X_train.shape[1])
        hidden = Dense(30, activation="tanh")(inputs)
        hidden = Dense(2, activation="step_wise")(hidden)
        hidden = Dense(30, activation="tanh")(hidden)
        outputs = Dense(X_train.shape[1], activation="linear")(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["MAE"])
        model.summary()

        history = model.fit(
            X_train, X_train, validation_split=0.4, epochs=100, batch_size=50, verbose=1
        )
        return model, history

    def outliers(self, model, X_train, X_test):
        """
        Identifies ouliers using autoencoder.
        """
        x_train_pred = model.predict(X_train)
        x_test_pred = model.predict(X_test)
        train_mae_loss = np.mean(
            np.abs(x_train_pred - np.array(X_train)), axis=1
        ).reshape((-1))
        test_mae_loss = np.mean(np.abs(x_test_pred - np.array(X_test)), axis=1).reshape(
            (-1)
        )
        anomalies = (test_mae_loss >= 0.5 * np.max(train_mae_loss)).tolist()
        histogram = train_mae_loss.flatten()
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Indices of anomaly samples: ", np.where(anomalies))
        print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))
        x_train_pred = model.predict(X_train)

        histogram = train_mae_loss.flatten()
<<<<<<< HEAD
        plt.hist(histogram, label="MAE Loss")

        plt.title("Mean Absolute Error Loss")
=======
        plt.hist(histogram,
                                       label = 'MAE Loss')

        plt.title('Mean Absolute Error Loss')
>>>>>>> d04dd51d29dc855d99575c3ae3336e7f0ace3396
        plt.xlabel("Training MAE Loss (%)")
        plt.ylabel("Number of Samples")
        plt.show()

<<<<<<< HEAD
        # histogram1 = test_mae_loss.flatten()
        # plt.hist(histogram1,
        #                               label = 'MAE Loss')

        # plt.title('Mean Absolute Error Loss')
        # plt.xlabel("Test MAE Loss (%)")
        # plt.ylabel("Number of Samples")
=======

        #histogram1 = test_mae_loss.flatten()
        #plt.hist(histogram1,
        #                               label = 'MAE Loss')

        #plt.title('Mean Absolute Error Loss')
        #plt.xlabel("Test MAE Loss (%)")
        #plt.ylabel("Number of Samples")

>>>>>>> d04dd51d29dc855d99575c3ae3336e7f0ace3396

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
    out = Outliers()
    model, history = out.make_autoencoder(X_train)
    out.outliers(model, X_train, X_test)
