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
    def __init__(self, dataframe):
        """
        Constructor.
        """
        self.dataframe = dataframe

        # Divide the dataset in train, validation and test in a static way
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataframe, self.dataframe['AGE_AT_SCAN'], test_size=0.3, random_state=14)


    def make_model(self):
        """
        """
        inputs = Input(shape=(424))
        hidden = Dense(128, activation ='relu')(inputs)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        outputs = Dense(1, activation ='linear')(hidden)

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
        Autoenoder trained comparing the output vector with the input features
        using the Mean Squared Error (MSE)  loss function.
        git"""
        inputs = Input(shape=(424))
        hidden = Dense(30, activation ='tanh')(inputs)
        hidden = Dense(2, activation ='sigmoid')(hidden) #this should be a stepwise function
        hidden = Dense(30, activation ='tanh')(hidden)
        outputs = Dense(424, activation ='linear')(hidden)

        rnn = Model(inputs=inputs, outputs=outputs)
        rnn.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics=['MSE'])
        rnn.summary()
        rnn_hist = rnn.fit(self.X_train, self.y_train, validation_split = 0.4, epochs = 10, batch_size = 50, verbose = 0) #CHANGE HERE# Trying increasing number of epochs and changing batch size


        plt.plot(rnn_hist.history["val_loss"],label = 'val')
        plt.plot(rnn_hist.history["loss"],label = 'train')
        plt.legend()
        plt.show()

        return rnn

if __name__ == "__main__":
    deep = Deep("data/FS_features_ABIDE_males.csv")
    deep.make_model().summary()
