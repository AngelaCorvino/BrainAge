from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
          model: the model architecture (Tensorflow Object).

        """
        inputs = Input(shape=(424))
        hidden = Dense(128, activation ='relu')(inputs)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        outputs = Dense(1, activation ='linear')(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['MSE'])
        model.summary()

        return model

    def make_autoencoder(self):
        """
        Autoenoder trained comparing the output vector with the input features
        using the Mean Squared Error (MSE)  loss function.
        Returns:
          model: the model architecture (Tensorflow Object).
          model_summary: a summary of the model's architecture.

        """
        inputs = Input(shape=(424))
        hidden = Dense(30, activation ='tanh')(inputs)
        hidden = Dense(2, activation ='sigmoid')(hidden) #this should be a stepwise function
        hidden = Dense(30, activation ='tanh')(hidden)
        outputs = Dense(424, activation ='linear')(hidden)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics=['MSE'])
        model.summary()
        return model

    def model_fit(self, model):
        '''
        This function  trains the model on the x_train data.
        Arguments:
         model: the model architecture (Tensorflow Object).
        Returns:
          model: the trained model.
          history: a summary of how the model trained (training error, validation error).
         '''

        history = model.fit(self.X_train, self.y_train, validation_split = 0.4,
        epochs = 10, batch_size = 50, verbose = 0) #CHANGE HERE# Trying increasing number of epochs and changing batch size


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
        training_validation_loss = pd.DataFrame.from_dict(history.history, orient='columns')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = training_validation_loss.index, y = training_validation_loss["loss"].round(6),
                           mode = 'lines',
                           name = 'Training Loss',
                           connectgaps=True))
        fig.add_trace(go.Scatter(x = training_validation_loss.index, y = training_validation_loss["val_loss"].round(6),
                           mode = 'lines',
                           name = 'Validation Loss',
                           connectgaps=True))

        fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title="Epoch",
        yaxis_title="Loss",
        font=dict(
        family="Arial",
        size=11,
        color="#7f7f7f"
        ))
        return fig.show()


#if __name__ == "__main__":
