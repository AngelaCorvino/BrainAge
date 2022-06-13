from keras.layers import Dense, Dropout, Input, Flatten
from keras.models import Model, load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class DeepRegression():
    """

    Class describing deep regression model.

    Ordinary least squares Linear Regression.
    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.
    Parameters
    ----------
     plot_loss : bool, default=True
        When set to ``True``, plots the training and validation loss curves of the trained model

    Attributes
    ----------
    model 



    """

    def __init__(
        self,
        epochs,
        plot_loss=True,
    ):
        self.epochs = epochs
        self.plot_loss = plot_loss


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
        inputs = Input(shape=(425))
        hidden = Dense(128, activation ='relu')(inputs)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        hidden = Dense(12, activation ='relu')(hidden)
        outputs = Dense(1, activation ='linear')(hidden)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss = 'mean_absolute_error', optimizer = 'adam', metrics=['MAE'])
        self.model.summary()
        history = self.model.fit(X, y, validation_split = 0.3, epochs = self.epochs, batch_size = 50, verbose = 0)
        return self







        if self.plot_loss==True:
            '''
            This parameter allows to plot the training and validation loss curves of the trained model,
            enabling visual diagnosis of underfitting (bias) or overfitting (variance).


            Returns
            -------
            fig: a visual representation of the model's training loss and validation
            loss curves.
            '''

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


        return self
