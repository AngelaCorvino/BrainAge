from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from features import Utilities

class Deep:
    """
    Class describing deep regression model.
    Parameters
    ----------
    file_url : string-like
        The path that point to the data.

    """
    def __init__(self, file_url):
        """
        Constructor.
        """
        self.file_url = file_url
        self.util = Utilities(file_url)
        (self.df_AS, self.df_TD) = self.util.file_split()
        self.df_TD = self.df_TD.drop(['Site', 'FILE_ID'], axis = 1)
        # Divide the dataset in train, validation and test in a static way
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_TD, self.df_TD['AGE_AT_SCAN'], test_size=0.3, random_state=14)
        print(self.X_train.shape)

    def make_model(self):
        """
        """
        inputs = Input(shape=(424))
        hidden = Dense(128, activation ='relu')(inputs)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        outputs = Dense(1, activation ='sigmoid')(hidden)

        deepmodel = Model(inputs=inputs, outputs=outputs)
        deepmodel.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['MSE'])
        deepmodel.summary()
        deephistory = deepmodel.fit(self.X_train, self.y_train, validation_split = 0.4, epochs = 1000, batch_size = 50, verbose = 0) #CHANGE HERE# Trying increasing number of epochs and changing batch size

        plt.plot(deephistory.history["val_loss"],label = 'val')
        plt.plot(deephistory.history["loss"],label = 'train')
        plt.legend()
        plt.show()
        # plt.semilogy(deephistory.history['MSE'])
        # plt.semilogy(deephistory.history['val_MSE'])
        # plt.show()
        return deepmodel
        
    def make_autoencoder(self):
        """
        For our application, we adapted the autoencoder described in 'S. Hawkins, H. He, G. Williams, R. Baxter, Outlier detection using repli-
cator neural networks' to our data, building a symmetric, four-linear-layer network with N = 424 (the number of features under examination). The three inner layers have 30, 2 and 30 neurons respectively, their activation functions are a hyperbolic tangent, a step-wise function and a hyperbolic tangent again. The fourth layer generates an output with the same dimensions as the input and a sigmoid filter maps the output into the final vector. We trained the autoencoder comparing the output vector with the input features using the Mean Squared Error (MSE) as the loss function.
        """
        inputs = Input(shape=(424))
        hidden = Dense(30, activation ='tanh')(inputs)
        hidden = Dense(2, activation ='sigmoid')(hidden) #this should be a stepwise function
        hidden = Dense(30, activation ='tanh')(hidden)
        outputs = Dense(424, activation ='sigmoid')(hidden)
        
        rnn = Model(inputs=inputs, outputs=outputs)
        rnn.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics=['MSE'])
        rnn.summary()
        rnn_hist = rnn.fit(self.X_train, self.y_train, validation_split = 0.4, epochs = 1000, batch_size = 50, verbose = 0) #CHANGE HERE# Trying increasing number of epochs and changing batch size
        
        plt.plot(rnn_hist.history["val_loss"],label = 'val')
        plt.plot(rnn_hist.history["loss"],label = 'train')
        plt.legend()
        plt.show()
        
        return rnn

if __name__ == "__main__":
    deep = Deep("data/FS_features_ABIDE_males.csv")
    deep.make_model().summary()
