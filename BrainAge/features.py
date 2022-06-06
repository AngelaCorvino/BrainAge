import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from neuroHarmonize import harmonizationLearn
from neuroCombat import neuroCombat

import statsmodels.api as sm

class Preprocessing:
    """
    Class containing functions for data

     Parameters
    ----------
    file_url : string-like
        The string containing data adress to be passed to Preprocessing.
    """

    def __init__(self):
        """
        Initialize the class.
        """

    def __call__(self, df):
        """
        What happens when you call an istance as a function
        """
        self.add_features(df)
        self.age_binning(df)
        self.site_binning(df)
        (df_AS, df_TD) = self.file_split(df)
        features, X, y = self.feature_selection(df_TD)
        print(features)
        prep.plot_histogram(df, 'AGE_AT_SCAN')
        return

    #def __str__(self):
    #    return "The dataset has {} size\n{} shape \nand these are the first 5 rows\n{}\n".format(df.size, df.shape, df.head(5))

    def file_reader(self, file_url):
        """
        Read data features from url and return them in a dataframe
        """
        df = pd.read_csv(file_url, sep = ";")
        return df

    def add_features(self, dataframe):
        """
        Add columns with derived features
        """
        dataframe['TotalWhiteVol'] = dataframe.lhCerebralWhiteMatterVol + dataframe.rhCerebralWhiteMatterVol
        dataframe['SITE'] = dataframe.FILE_ID.apply(lambda x: x.split('_')[0])
        return

    def age_binning(self, dataframe):
        """
        Create a column with AGE_AT_SCAN binning and attach to dataframe
        """
        dataframe['AGE_CLASS'] = pd.cut(dataframe.AGE_AT_SCAN, 6, labels = [x for x in range(6)])
        return dataframe['AGE_CLASS']

    def site_binning(self, dataframe):
        """
        Create a map  where SITE is binned and then merge it withe dataframe
        """
        try :
            grouping_lists=['Caltech','CMU','KKI','Leuven','MaxMun','NYU',
        'OHSU','Olin','Pitt','SBL','Stanford','Trinity','UCLA', 'UM','USM','Yale']
            labels=[x for x in range(16)]
            maps = (pd.DataFrame({'SITE_CLASS': labels, 'SITE': grouping_lists})
            .explode('SITE')
            .reset_index(drop=True))

            dataframe = dataframe.merge(maps, on = 'SITE', how='left').fillna("Other")
        except KeyError:
             print("Column SITE does not exist")

        return dataframe

    def file_split(self, dataframe):
        """
        Split dataframe in healthy (control) and autistic subjects groups
        """
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
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


    def neuro_harmonization(self, dataframe, confounder="SITE", covariate1="AGE_AT_SCAN"):
        """
        Harmonize dataset with neuroHarmonize model.
        1-Load your data and all numeric covariates
        2-run harmonization and store the adjusted data
        """
        covars = dataframe[[confounder, covariate1]]
        dataframe = np.array(dataframe)
        my_model, df_neuroharmonized,s_data = harmonizationLearn(dataframe, covars,return_s_data=True)

        return df_neuroharmonized

    def com_harmonization(self, dataframe, confounder="SITE_CLASS", covariate="AGE_AT_SCAN"):
        """
        Harmonize dataset with ComBat model
        """
        dataframe=  dataframe.drop([ 'SITE'], axis = 1)
        df_combat = neuroCombat(
            dat = dataframe.transpose(),
            covars = dataframe[[confounder, covariate]],
            batch_col = confounder,
        )["data"]
        df_combatharmonized = df_combat.transpose()
        #df_TDharmonized = self.df_TD[self.features]
        #df_TDharmonized.loc[:, (self.features)] = df_combat.transpose()
        # the following line has to be inseting in the next function
        #X_train, X_test, y_train, y_test = train_test_split(
        #    df_TDharmonized, self.df_TD["AGE_AT_SCAN"], test_size=0.3
        #)
        return df_combatharmonized


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
        return  listoffeatures, X, y


if __name__ == "__main__":
    prep = Preprocessing()
    df = prep.file_reader("data/FS_features_ABIDE_males.csv")
    prep(df)
    prep.com_harmonization(df, confounder="SITE_CLASS", covariate="AGE_AT_SCAN")
