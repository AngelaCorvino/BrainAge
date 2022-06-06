from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


    def make_MLP(self):
        """
        This function uses   generates an MLP model.
        Arguments:

          Returns:

          model: the trained model.
          history: a summary of how the model trained (training error, validation error).

        """
        inputs = Input(shape=(425))
        hidden = Dense(128, activation ='relu')(inputs)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        outputs = Dense(1, activation ='linear')(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['MAE'])
        model.summary()


        history = model.fit(self.X_train, self.y_train, validation_split = 0.4, epochs = 100, batch_size = 50, verbose = 0)

        return model, history

    def make_autoencoder(self):
        """
        Autoenoder trained comparing the output vector with the input features
        using the Mean Squared Error (MSE)  loss function.
        Returns:
        model: the trained model.
        history: a summary of how the model trained (training error, validation error).

        """
        inputs = Input(shape=(425))
        hidden = Dense(30, activation ='tanh')(inputs)
        hidden = Dense(2, activation ='sigmoid')(hidden) #this should be a stepwise function
        hidden = Dense(30, activation ='tanh')(hidden)
        outputs = Dense(425, activation ='linear')(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['MAE'])
        model.summary()


        history = model.fit(self.X_train, self.X_train, validation_split = 0.4,
        epochs = 100, batch_size = 50, verbose = 1) #CHANGE HERE# Trying increasing number of epochs and changing batch size


        return model, history

    def plot_training_validation_loss(self,history):
        '''
        This function plots the training and validation loss curves of the trained model,
        enabling visual diagnosis of underfitting (bias) or overfitting (variance).
        Arguments:
          history

        Returns:
          fig: a visual representation of the model's training loss and validation
          loss curves.
         '''
        #      plt.plot(rnn_hist.history["val_loss"],label = 'val')
        #      plt.plot(rnn_hist.history["loss"],label = 'train')
        #      plt.legend()
        #      plt.show()
        #
        training_validation_loss = pd.DataFrame.from_dict(history.history, orient='columns')


        plt.scatter(training_validation_loss.index,training_validation_loss["loss"],
                           marker='.',
                           label= 'Training Loss',
                           )
        plt.scatter(training_validation_loss.index,training_validation_loss["val_loss"],
                        marker='.',
                        label = 'Validation Loss',
                            )


        plt.title('Training and Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        return plt.show()

    def reconstruction_error(self,model):
        """
        This function calculates the reconstruction error and displays a histogram of
        the training mean absolute error.
        Arguments:
        model: the trained  model
          x_train: 3D data to be used in model training (dataframe).
          Returns:
          fig: a visual representation of the training MAE distribution.
        """


        x_train_pred = model.predict(self.X_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - np.array(self.X_train)), axis = 1)
        histogram = train_mae_loss.flatten()
        plt.hist(histogram,
                                      label = 'MAE Loss')

        plt.title('Mean Absolute Error Loss')
        plt.xlabel("Training MAE Loss (%)")
        plt.ylabel("Number of Samples")


        print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))

        return plt.show()



#if __name__ == "__main__":
