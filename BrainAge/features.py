import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from neuroHarmonize import harmonizationLearn
from neuroCombat import neuroCombat

pd.set_option("display.max_rows", None)



class Preprocessing:
    """
    Class containing functions for data
    """

    def __call__(self, dataframe, harmonize_option, plot_option=True):
        """
        Allows to you call an istance as a function.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of raw data  to be passed to the preprocessing class.
        harmonize_option : string-like
                           String containing one of the options: 'raw', 'combat', 'neuro'.
        plot_option : boolean
                      If True shows some plots of data. Default is True

        Returns
        -------

        dataframe_harmonized : dataframe-like
                               Dataframe containing harmonized data.
        """
        self.add_features(dataframe)
        self.add_age_binning(dataframe)
        # PLOTTING DATA
        if plot_option == True:
            self.plot_boxplot(dataframe, "SITE", "AGE_AT_SCAN")
            self.plot_histogram(dataframe, "AGE_AT_SCAN")
        if harmonize_option == "not_normalized":
            dataframe = dataframe.drop(["FILE_ID", "SITE"], axis=1)
            return dataframe
        self.self_normalize(dataframe)
        # HARMONIZING DATA
        if harmonize_option == "raw":
            dataframe = dataframe.drop(["FILE_ID", "SITE"], axis=1)
            return dataframe

        elif harmonize_option == "combat":
            dataframe = self.add_site_binning(dataframe)
            try:
                assert (
                    np.sum(np.sum(dataframe.isna())) == 0
                ), "There are NaN values in the dataframe!"
                assert (
                    np.sum(np.sum(dataframe.isnull())) == 0
                ), "There are Null values in the dataframe!"
            except AssertionError as msg:
                print(msg)
            dataframe_combat = self.com_harmonize(
                dataframe.drop(["FILE_ID", "SITE"], axis=1),
                confounder="SITE_CLASS",
                covariate="AGE_AT_SCAN",
            )
            dataframe_combat = dataframe_combat.drop(["SITE_CLASS"], axis=1)

            return dataframe_combat
        elif harmonize_option == "neuro":
            dataframe_neuro = self.neuro_harmonize(
                dataframe, confounder="SITE", covariate1="AGE_AT_SCAN"
            )

            return dataframe_neuro

    # def __str__(self):
    #    return "The dataset has {} size\n{} shape \nand these are the first 5 rows\n{}\n".format(df.size, df.shape, df.head(5))

    def read_file(self, file_url):
        """
        Reads data in .csv file from url and returns them in a dataframe.

        Parameters
        ----------

        file_url : string-like
                   The string containing data adress to be passed to
                   Preprocessing.

        Returns
        -------

        dataframe : dataframe-like
                    The dataframe of raw data.
        """
        dataframe = pd.read_csv(file_url, sep=";")
        return dataframe
        
    def split_file(self, dataframe):
        """
        Splits dataframe in ASD cases and controls.
        
        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be split.

        Returns
        -------

        df_AS : dataframe-like
                The dataframe containing ASD cases.
        
        df_TD : dataframe-like
                The dataframe containing controls.

        """
        df_AS = dataframe.loc[dataframe.DX_GROUP == 1]
        df_TD = dataframe.loc[dataframe.DX_GROUP == -1]
        return df_AS, df_TD

    def add_features(self, dataframe):
        """
        Adds columns with derived features to dataframe.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of raw data  to be passed to the function.
        """
        dataframe["TotalWhiteVol"] = (
            dataframe.lhCerebralWhiteMatterVol + dataframe.rhCerebralWhiteMatterVol
        )
        dataframe["SITE"] = dataframe.FILE_ID.apply(lambda x: x.split("_")[0])
        dataframe = dataframe.drop(["FILE_ID"], axis=1)


    def add_age_binning(self, dataframe):
        """
        Creates a column called AGE_CLASS with AGE_AT_SCAN binning and attaches it to dataframe.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        """
        bins = 6
        dataframe["AGE_CLASS"] = pd.qcut(
            dataframe.AGE_AT_SCAN, bins, labels=[x for x in range(1, bins + 1)]
        )


    def add_site_binning(
        self, dataframe
    ):  # capire se si può fare senza return come add_age_binning
        """
        Creates a map  where SITE  is binned in the column SITE_CLASS and then merges it with the dataframe.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        """
        try:
            sites = (
                dataframe["SITE"]
                .value_counts(dropna=False, sort=False)
                .keys()
                .to_list()
            )
            maps = (
                pd.DataFrame(
                    {"SITE_CLASS": [x for x in range(1, len(sites) + 1)], "SITE": sites}
                )
                .explode("SITE")
                .reset_index(drop=True)
            )
            dataframe = dataframe.join(maps.set_index("SITE"), on="SITE")
        except KeyError:
            print("Column SITE does not exist")
        return dataframe

    def plot_histogram(self, dataframe, feature):
        """
        Plots histogram of a given feature on the indicated dataframe, masking values <0.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        feature : string-like
                  The feature to plot the histogram of.
        """
        dataframe[dataframe.loc[:, feature] > 0].hist([feature])
        plt.show()

    def plot_boxplot(self, dataframe, featurex, featurey):
        """
        Plots boxplot of featurey by featurex.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        featurex : string-like
                   The feature in the x-axis of the boxplot.
        featurey : string-like
                   The feature in the y-axis of the boxplot.
        """
        sns_boxplot = sns.boxplot(x=featurex, y=featurey, data=dataframe)
        sns_boxplot.set_xticklabels(labels=sns_boxplot.get_xticklabels(), rotation=50)
        sns_boxplot.grid()
        sns_boxplot.set_title("Box plot of " + featurey + " by " + featurex)
        sns_boxplot.set_ylabel(featurey)
        plt.show()


    def self_normalize(self, dataframe):
        """
        Makes self normalization on data.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.

        """
        # Surface
        column_list = dataframe.loc[
            :, ["SurfArea" in i for i in dataframe.columns]
        ].columns.tolist()
        dataframe.loc[:, ["SurfArea" in i for i in dataframe.columns]] = dataframe.loc[
            :, ["SurfArea" in i for i in dataframe.columns]
        ].divide(dataframe[column_list].sum(axis=1), axis=0)

        # Thickness
        dataframe.loc[:, ["ThickAvg" in i for i in dataframe.columns]] = dataframe.loc[
            :, ["ThickAvg" in i for i in dataframe.columns]
        ].divide(
            (dataframe["lh_MeanThickness"] + dataframe["rh_MeanThickness"]), axis=0
        )
        # print(dataframe.iloc[:,:20])

        # Volume
        dataframe.loc[:, ["Vol" in i for i in dataframe.columns]] = dataframe.loc[
            :, ["Vol" in i for i in dataframe.columns]
        ].divide((dataframe["TotalGrayVol"] + dataframe["TotalWhiteVol"]), axis=0)

        return

    def neuro_harmonize(self, dataframe, confounder="SITE", covariate1="AGE_AT_SCAN"):
        """
        Harmonize dataset with neuroHarmonize model:
        1-Load your data and all numeric covariates
        2-Run harmonization and store the adjusted data.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        confounder : string-like
                     Feature to be accounted as confounder. Default is 'SITE'
        covariate1 : string-like

        Returns
        -------

        dataframe_harmonized: dataframe-like
                              The dataframe containing harmonized data
        """
        try:
            covars = dataframe[[confounder, covariate1]]
            dataframe = dataframe.drop(["FILE_ID", "SITE"], axis=1)
            # dataframe=dataframe.astype(np.float)
            my_model, array_neuro_harmonized, s_data = harmonizationLearn(
                np.array(dataframe), covars, return_s_data=True
            )
            df_neuro_harmonized = pd.DataFrame(array_neuro_harmonized)
            df_neuro_harmonized.columns = dataframe.keys()
            df_neuro_harmonized[
                ["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FIQ"]
            ] = dataframe[["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FIQ"]]
        except RuntimeWarning:
            print("How can we solve this?")
        return df_neuro_harmonized

    def com_harmonize(
        self, dataframe, confounder="SITE_CLASS", covariate="AGE_AT_SCAN"
    ):
        """
        Harmonize dataset with ComBat model

        Parameters
        ----------

        dataframe : dataframe-like
                    Dataframe containing neuroimaging data to correct with shape = (samples, features) e.g. cortical thickness measurements, image voxels, etc
        confounder : string-like
                     Categorical feature to be accounted as confounder, which indicates batch (scanner) column name in covars (e.g. "scanner") Default 'SITE_CLASS'.
        covariate : string-like
                    Contains the batch/scanner covariate as well as additional covariates (optional) that should be preserved during harmonization. Default is 'AGE_AT_SCAN'.

        Returns
        -------

        dataframe_harmonized: dataframe-like
                              The dataframe containing harmonized data
        """
        array_combat_harmonized = neuroCombat(
            dat=dataframe.transpose(),
            covars=dataframe[[confounder, covariate]],
            batch_col=confounder,
        )["data"]
        df_combat_harmonized = pd.DataFrame(array_combat_harmonized.transpose())
        df_combat_harmonized.columns = dataframe.keys()
        df_combat_harmonized[
            ["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FIQ"]
        ] = dataframe[["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FIQ"]]

        return df_combat_harmonized

    def feature_selection(self, dataframe, feature="AGE_AT_SCAN", plot_heatmap=False):
        """
        Gives a list of the feature whose correlation with the given feature is higher than 0.5.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of data to be passed to the function.
        feature : string-like
                  Target feature on which to compute correlation.
        plot_heatmap : boolean
                       If True, show heatmap of data correlation with feature. Default is False.

        Returns
        -------

        listoffeatures : list
                         List of selected features.
        X : dataframe-like
        y : series-like?
        """
        agecorr = dataframe.corr()[feature]
        listoffeatures = agecorr[np.abs(agecorr) > 0.5].keys()
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
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    df1 = prep(df, "raw", plot_option=False)
    df2 = prep(df, "neuro", plot_option=False)
    df3 = prep(df, "combat", plot_option=False)
    # print(df1)
    # print(df2)
    # print(df3)
