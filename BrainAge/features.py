import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Utilities:
    """
    Class containing functions for data

     Parameters
    ----------
    file_url : string-like
        The string containing data adress to be passed to Utilities .
    """

    def __init__(self, file_url,harmonization):
        """
        Initialize the class.
        """


        self.df = pd.read_csv(file_url, sep = ";")



        self.df = self.add_features()

        (self.df_AS, self.df_TD) = self.file_split()
        self.features,self.X,self.y=self.feature_selection()
        self.harmonization=harmonization

        if self.harmonization==True:
            self.df_TDharmonized=self.com_harmonization()
            self.df_TD=self.df_TDharmonized

    def __str__(self):
        return "The dataset has {} size\n{} shape \nand these are the first 5 rows\n{}\n".format(self.df.size, self.df.shape, self.df.head(5))



    def add_features(self):
        """
        Add columns with derived features
        """
        self.df['TotalWhiteVol'] = self.df.lhCerebralWhiteMatterVol + self.df.rhCerebralWhiteMatterVol
        self.df['Site'] = self.df.FILE_ID.apply(lambda x: x.split('_')[0])


    def add_binning(self, dataframe):
        """
        Add a column to control  with AGE_AT_SCAN binning
        """
        #self.df['AGE_CLASS'] = pd.cut(self.df.AGE_AT_SCAN, 6, labels = [x for x in range(6)])
        getattr(self, dataframe)['AGE_CLASS'] = pd.cut(getattr(self, dataframe).AGE_AT_SCAN, 4, labels = [x for x in range(4)])
        return getattr(self, dataframe).AGE_CLASS

    def file_split(self):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = self.df.loc[self.df.DX_GROUP == 1]
        df_TD = self.df.loc[self.df.DX_GROUP == -1]
        return df_AS, df_TD

    def plot_histogram(self, feature, control = True):
        """
        Plot histogram of a given feature on the indicated group, masking values <0
        """
        if control == True:
            self.df_TD[self.df_TD.loc[:, feature]>0].hist([feature])
        elif control == False:
            self.df_ASD[self.gdf_AS.loc[:, feature]>0].hist([feature])
        plt.show()
        return

    def plot_boxplot(self, featurex, featurey, control = True):
        """
        Boxplot of featurey by featurex
        """
        if control == True:
            sns_boxplot = sns.boxplot(x=featurex, y=featurey, data=self.df_TD)
        if control == False:
            sns_boxplot = sns.boxplot(x=featurex, y=featurey, data=self.df_AS)
        sns_boxplot.set_xticklabels(labels=sns_boxplot.get_xticklabels(), rotation=50)
        sns_boxplot.grid()
        sns_boxplot.set_title('Box plot of '+ featurey + 'by ' + featurex)
        sns_boxplot.set_ylabel(featurey)
        plt.show()
        return

    def com_harmonization(self, confounder="Site", covariate="AGE_AT_SCAN"):
        """
        Harmonize dataset with ComBat model
        """

        df_combat = neuroCombat(
            dat=self.df_TD.transpose(),
            covars=self.df_TD[[confounder, covariate]],
            batch_col=confounder,
        )["data"]

        df_TDharmonized = df_combat.transpose()
        #df_TDharmonized = self.df_TD[self.features]
        #df_TDharmonized.loc[:, (self.features)] = df_combat.transpose()
        # the following line has to be inseting in the next function
        #X_train, X_test, y_train, y_test = train_test_split(
        #    df_TDharmonized, self.df_TD["AGE_AT_SCAN"], test_size=0.3
        #)

        return df_TDharmonized

    def feature_selection(self, feature = 'AGE_AT_SCAN', plot_heatmap = False):
        """
        Gives a list of the feature whose correlation with the given feature is higher than 0.5
        """
        agecorr_TD = self.df_TD.corr()[feature] #we acces to the column relative to age
        listoffeatures = agecorr_TD[np.abs(agecorr_TD)>0.5].keys() #we decide to use the feautures whose correlation with age at scan is >0.5
        if plot_heatmap == True:
            df_TDrestricted = self.df_TD[listoffeatures]
            heatmap = sns.heatmap(df_TDrestricted.corr(), annot=True)
            plt.show()
        listoffeatures = listoffeatures.drop(feature)
        X = self.df_TD[listoffeatures]
        y = self.df_TD[feature]

        return listoffeatures,X,y




if __name__ == "__main__":
    util = Utilities("data/FS_features_ABIDE_males.csv")
    #util.plot_histogram('AGE_AT_SCAN')
    #util.plot_boxplot('Site', 'AGE_AT_SCAN', True)
