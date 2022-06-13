from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def step_wise(x):
    N = 4
    y = 1/2
    for j in range(1, N):
        print(j)
        y+=((1/(2*(N-1)))*(np.tanh(100*(x-(j/N)))))
    return y

class Deep:
    """Class describing deep regression model.

    Parameters
    ----------
    dataframe : type
        Description of parameter `dataframe`.

    Attributes
    ----------
    X_train : type
        Description of attribute `X_train`.
    X_test : type
        Description of attribute `X_test`.
    y_train : type
        Description of attribute `y_train`.
    y_test : type
        Description of attribute `y_test`.
    dataframe

    """

    def __init__(self, dataframe):
        """
        Constructor.
        """
        self.dataframe = dataframe
        # Divide the dataset in train, validation and test in a static way
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataframe, self.dataframe['AGE_AT_SCAN'], test_size=0.3, random_state=14)


    def make_autoencoder(self):
        """Autoencoder trained comparing the output vector with the input features, using the Mean Squared Error (MSE) as loss function.

        Returns
        -------
        type
            Description of returned object.
        model : the trained model.
        history : a summary of how the model trained (training error, validation error).

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

    def outliers(self,model):
        """

        """
        x_train_pred = model.predict(self.X_train)
        x_test_pred = model.predict(self.X_test)
        train_mae_loss = np.mean(np.abs(x_train_pred - np.array(self.X_train)), axis = 1).reshape((-1))
        test_mae_loss = np.mean(np.abs(x_test_pred - np.array(self.X_test)), axis = 1).reshape((-1))
        anomalies = (test_mae_loss >= np.max(train_mae_loss)).tolist()
        histogram = train_mae_loss.flatten()
        print("Number of anomaly samples: ", np.sum(anomalies))
        print("Indices of anomaly samples: ", np.where(anomalies))

        print("Reconstruction error threshold: {} ".format(np.max(train_mae_loss)))

        return


#if __name__ == "__main__":
