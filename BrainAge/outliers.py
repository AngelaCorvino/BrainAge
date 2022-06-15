# pylint: disable=invalid-name, redefined-outer-name
import scipy.stats as stats
import scipy.optimize
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Activation
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

def gaussian(x, x0, sigma, a):
    return a*(1/np.sqrt(np.pi*sigma**2))*np.exp(-(x-x0)**2/(sigma**2))
    
def sumgaussian(x, x0, x1, sigma0, sigma1, a, b):
    return a*gaussian(x, x0, sigma0) + b*gaussian(x, x1, sigma1)

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
        x_train_pred = self.model.predict(self.X_train)
        x_test_pred = self.model.predict(self.X_test)

        train_mae_loss = np.mean(np.abs(x_train_pred - np.array(self.X_train)), axis=1).reshape((-1))
        train_mae_loss /= np.abs(stats.zscore(train_mae_loss))
        # train_mse_loss = np.square(np.subtract(np.array(self.X_train), x_train_pred))
        # print(train_mse_loss)
        # train_mse_loss=train_mse_loss.mean(axis = 1)
        # print(train_mse_loss)
        test_mae_loss = np.mean(np.abs(x_test_pred - np.array(self.X_test)), axis=1).reshape((-1))
        test_mae_loss /= np.abs(stats.zscore(test_mae_loss))
        #test_mse_loss = (np.square(np.subtract(np.array(self.X_test), x_test_pred)).mean(axis = 1))
        initial_outliers = (test_mae_loss >= np.max(train_mae_loss)).tolist()
        print("Number of outlier samples: ", np.sum(initial_outliers))
        print("Indices of outlier samples: ", np.where(initial_outliers))
        
        nbins = 500
        
        #Plot_train_variables
        d_0=(train_mae_loss.max()-train_mae_loss.min())/nbins
        xdiscrete_0=np.linspace(train_mae_loss.min()+d_0/2, train_mae_loss.max()-d_0/2, nbins)
        xcont_0=np.linspace(train_mae_loss.min(), train_mae_loss.max(), 1000)
        #Plot_train
        plt.figure(1)
        n_0, bins_0, _ = plt.hist(x = train_mae_loss, bins = nbins, color='lightskyblue', label = 'Sample counts')
        plt.title('Mean Absolute Error Loss')
        plt.xlabel("Train MSE Loss (Z-score)")
        plt.ylabel("Number of Samples")       
        #fit_train
        fit_0, fitCov_0 = scipy.optimize.curve_fit(gaussian, xdiscrete_0, n_0)
        fit_err_0 = np.sqrt(abs(np.diag(fitCov_0)))
        print(f' Train fit parameters: \n x_0 = {fit_0[0] : .3f} +-{fit_err_0[0] : .3f}\n sigma_0 = {fit_0[1] : .3f} +-{fit_err_0[1] : .3f}\n A = {fit_0[2] : .3f} +-{fit_err_0[2] : .3f}\n')
        #Plot_fit_train
        plt.plot(xcont_0, gaussian(xcont_0, *fit_0), color = 'red', linewidth=1, label='fit')
        plt.show()

        #Plot_test_variables
        d_1=(test_mae_loss.max()-test_mae_loss.min())/nbins
        xdiscrete_1=np.linspace(test_mae_loss.min()+d_1/2, test_mae_loss.max()-d_1/2, nbins)
        xcont_1=np.linspace(test_mae_loss.min(), test_mae_loss.max(), 1000)
        #Plot_test
        plt.figure(2)
        n_1, bins_1, _ = plt.hist(x = test_mae_loss, bins = nbins, color='lightskyblue', label = 'Sample counts')
        plt.title('Mean Absolute Error Loss')
        plt.xlabel("Test MSE Loss (Z-score)")
        plt.ylabel("Number of Samples")
        
        #fit_test
        fit_1, fitCov_1 = scipy.optimize.curve_fit(gaussian, xdiscrete_1, n_1)
        fit_err_1 = np.sqrt(abs(np.diag(fitCov_1)))
        print(f' Test fit parameters: \n x_0 = {fit_1[0] : .3f} +-{fit_err_1[0] : .3f}\n sigma_0 = {fit_1[1] : .3f} +-{fit_err_1[1] : .3f}\n A = {fit_1[2] : .3f} +-{fit_err_1[2] : .3f}\n')
        #print(f' First fit parameters: \n x_0 = {fit[0] : .3f} +-{fit_err[0] : .3f}\n x_1 = {fit[1] : .3f} +-{fit_err[1] : .3f}\n sigma_0 = {fit[2] : .3f} +-{fit_err[2] : .3f}\n sigma_1 = {fit[3] : .3f} +-{fit_err[3] : .3f}\n A = {fit[4] : .3f} +-{fit_err[4] : .3f}\n B = {fit[5] : .3f} +-{fit_err[5] : .3f}\n')
        #Plot_fit_test
        plt.plot(xcont_1, gaussian(xcont_1, *fit_1), color = 'red', linewidth=1, label='fit')
        plt.show()
        
        def condition(x, fit_1):
            if x >= (fit_1[0] + 3*fit_1[1]) : return True
            if x <= (fit_1[0] - 3*fit_1[1]) : return True
            else : return False
            
        final_outliers = [condition(x, fit_1) for x in test_mae_loss]
        
        print("Number of final outlier samples: ", np.sum(final_outliers))
        print("Indices of final outlier samples: ", np.where(final_outliers))
        
        # print("Reconstruction MAE error threshold: {} ".format(np.max(train_mae_loss)))
        # print("Reconstruction MSE error threshold: {} ".format(np.max(train_mse_loss)))
        return


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
