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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_TD, self.df_TD['AGE_AT_SCAN'], test_size=0.7, random_state=14)
        print(self.X_train.shape)

    def make_model(self):
        """
        """
        inputs = Input(shape=(424))
        hidden = Dense(12, activation='relu')(inputs)
        hidden = Dense(12, activation='relu')(hidden)
        hidden = Dense(12, activation='relu')(hidden)
        hidden = Dense(12, activation='relu')(hidden)
        outputs = Dense(1, activation='sigmoid')(hidden)

        deepmodel = Model(inputs=inputs, outputs=outputs)
        deepmodel.compile(loss='binary_crossentropy', optimizer='adam')
        deepmodel.summary()
        deephistory=deepmodel.fit(self.X_train,self.y_train,validation_split=0.5,epochs=2500,batch_size=128,verbose=0) #CHANGE HERE# Trying increasing number of epochs and changing batch size

        plt.plot(history.history["val_loss"])
        plt.plot(history.history["loss"])
        plt.plot(deephistory.history["val_loss"])
        plt.plot(deephistory.history["loss"])
        plt.show()
        return model

if __name__ == "__main__":
    deep=Deep("data/FS_features_ABIDE_males.csv")
    deep.make_model().summary()
