import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Preprocessing:
    """
    Class containing functions for data

     Parameters
    ----------
    file_url : string-like
        The string containing data adress to be passed to Preprocessing.
    df : dataframe containing data to preprocess.
    """

    def __init__(self):
        """
        Initialize the class.
        """

    #def __str__(self):
    #    return "The dataset has {} size\n{} shape \nand these are the first 5 rows\n{}\n".format(df.size, df.shape, df.head(5))

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



    def add_features(self, df):
        """
        Add columns with derived features
        """
        df['TotalWhiteVol'] = df.lhCerebralWhiteMatterVol + df.rhCerebralWhiteMatterVol
        df['Site'] = df.FILE_ID.apply(lambda x: x.split('_')[0])
        return


    def add_binning(self, dataframe):
        """
        Add a column to control  with AGE_AT_SCAN binning
        """
        #self.df['AGE_CLASS'] = pd.cut(self.df.AGE_AT_SCAN, 6, labels = [x for x in range(6)])
        dataframe['AGE_CLASS'] = pd.cut(dataframe.AGE_AT_SCAN, 6, labels = [x for x in range(6)])
        return dataframe['AGE_CLASS']

    def file_split(self, df):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = df.loc[df.DX_GROUP == 1]
        df_TD = df.loc[df.DX_GROUP == -1]
        return df_AS, df_TD

    def plot_histogram(self, dataframe, feature):
        """
        Plot histogram of a given feature on the indicated group, masking values <0
        """
        dataframe[dataframe.loc[:, feature]>0].hist([feature])
        plt.show()
        return

    def plot_boxplot(self, dataframe, featurex, featurey):
        """
        Boxplot of featurey by featurex
        """
        sns_boxplot = sns.boxplot(x = featurex, y = featurey, data = dataframe)
        sns_boxplot.set_xticklabels(labels = sns_boxplot.get_xticklabels(), rotation=50)
        sns_boxplot.grid()
        sns_boxplot.set_title('Box plot of '+ featurey + ' by ' + featurex)
        sns_boxplot.set_ylabel(featurey)
        plt.show()
        return

    def com_harmonization(self, dataframe, confounder="Site", covariate="AGE_AT_SCAN"):
        """
        Harmonize dataset with ComBat model
        """
        df_combat = neuroCombat(
            dat = dataframe.transpose(),
            covars = dataframe[[confounder, covariate]],
            batch_col = confounder,
        )["data"]
        dataframe_harmonized = df_combat.transpose()
        #df_TDharmonized = self.df_TD[self.features]
        #df_TDharmonized.loc[:, (self.features)] = df_combat.transpose()
        # the following line has to be inseting in the next function
        #X_train, X_test, y_train, y_test = train_test_split(
        #    df_TDharmonized, self.df_TD["AGE_AT_SCAN"], test_size=0.3
        #)
        return dataframe_harmonized

    def feature_selection(self, dataframe, feature = 'AGE_AT_SCAN', plot_heatmap = False):
        """
        Gives a list of the feature whose correlation with the given feature is higher than 0.5
        """
        agecorr = dataframe.corr()[feature] #we acces to the column relative to age
        listoffeatures = agecorr[np.abs(agecorr)>0.5].keys() #we decide to use the feautures whose correlation with age at scan is >0.5
        if plot_heatmap == True:
            dataframe_restricted = dataframe[listoffeatures]
            heatmap = sns.heatmap(dataframe_restricted.corr(), annot=True)
            plt.show()
        listoffeatures = listoffeatures.drop(feature)
        X = dataframe[listoffeatures]
        y = dataframe[feature]

        return listoffeatures, X, y


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.file_reader("data/FS_features_ABIDE_males.csv")
    print(df.shape)
    prep.add_features(df)
    print(df.shape)
    prep.plot_histogram(df, 'AGE_AT_SCAN')
    (df_AS, df_TD) = prep.file_split(df)
    prep.plot_boxplot(df_TD, 'Site', 'AGE_AT_SCAN')
    prep.add_binning(df)
    print(df)
    features, X, y = prep.feature_selection(df_TD)
    print(features.shape)
    harmonization = False
    if harmonization == True:
        df_TD = prep.com_harmonization(df_TD)
        print(df_TD)
