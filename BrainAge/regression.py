import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from features import Utilities


class Regression():
    """
    Class describing regression model.
    Parameters
    ----------
    file_url : string-like
        The path that point to the data.

    """

    def __init__(self, file_url,features=0):
        """
        Constructor.
        """
        self.file_url = file_url
        self.util = Utilities(file_url)
        (self.df_AS, self.df_TD) = self.util.file_split()
        self.features=features


    def feature_selection(self,heatmap=True):
        """
        This function select the feautures according to correlation
        to AGE AT SCAN. The it removes AGE at SCAN from the list.
        """
        agecorr_TD=self.df_TD.corr()['AGE_AT_SCAN'] #we acces to the column relative to age
        self.features=agecorr_TD[np.abs(agecorr_TD)>0.5].keys()
        if heatmap==True:
            sns.heatmap(self.df_TD[self.features].corr(),annot=True)
            plt.show()
        self.features = self.features.drop('AGE_AT_SCAN')
        return self.features
