# pylint: disable=invalid-name, redefined-outer-name
"""
Module contains a class Preprocessing, with methods allowing to upload dataframe from file, add derived features, preprocess with some normalization options and harmonization options.
"""
import inspect
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from neuroHarmonize import harmonizationLearn
from neuroCombat import neuroCombat



class Preprocessing:
    """
    Class containing functions for data
    """

    def __call__(self, dataframe, prep_option, plot_option=True):
        """
        Allows to you call an istance as a function.

        Parameters
        ----------

        dataframe : dataframe-like
                    The dataframe of raw data  to be passed to the preprocessing class.
        prep_option : string-like
                           String containing one of the options: 'not_normalized', 'normalized', 'combat', 'neuro'.
        plot_option : boolean
                      If True shows some plots of data. Default is True

        Returns
        -------

        dataframe_harmonized : dataframe-like
                               Dataframe containing harmonized data.
        """
        self.add_TotalWhiteVol(dataframe)
        self.add_age_binning(dataframe)
        self.add_site(dataframe)
        dataframe = self.remove_FIQ(dataframe)
        # PLOTTING DATA
        if plot_option is True:
            self.plot_boxplot(dataframe, "SITE", "AGE_AT_SCAN")
            self.plot_histogram(dataframe, "AGE_AT_SCAN")
        # PROCESSING DATA
        if prep_option == "not_normalized":
            print("Dataframe is not normalised")
        elif prep_option == "normalized":
            self.self_normalize(dataframe)
            print("Dataframe is only normalized")
        elif prep_option == "combat_harmonized":
            self.self_normalize(dataframe)
            dataframe = self.add_site_binning(dataframe)
            try:
                assert (
                    np.sum(np.sum(dataframe.isna())) == 0
                ), "There are NaN values in the dataframe!"
            except AssertionError as msg:
                print(msg)
            dataframe_combat = self.com_harmonize(
                dataframe,
                confounder="SITE_CLASS",
                covariate="AGE_AT_SCAN",
            )
            dataframe_combat = dataframe_combat.drop(["SITE_CLASS"], axis=1)
            dataframe = dataframe_combat
            print("Dataframe is normalized and harmonized with NeuroCombat")

        elif prep_option == "neuro_harmonized":
            self.self_normalize(dataframe)
            dataframe_neuro = self.neuro_harmonize(
                dataframe,
                confounder="SITE",
                covariate1="AGE_AT_SCAN",
            )
            dataframe = dataframe_neuro
            print("Dataframe is normalized and harmonized with NeuroHarmonize")

        dataframe.drop(["FILE_ID"], axis=1, inplace=True)
        return dataframe

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

    def add_TotalWhiteVol(self, dataframe):
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

    def remove_FIQ(self, dataframe):
        """

        """
        dataframe = dataframe.drop(["FIQ"], axis = 1)
        return dataframe

    def add_site(self, dataframe):
        """
        Adds column with Site description.
        """
        dataframe["SITE"] = dataframe.FILE_ID.apply(lambda x: x.split("_")[0])

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
        Creates a map  where SITE  is binned in the column SITE_CLASS,
        then merges it with the dataframe.

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
        name = self.retrieve_name(dataframe)
        sns_boxplot = sns.boxplot(x=featurex, y=featurey, data=dataframe)
        sns_boxplot.set_xticklabels(labels=sns_boxplot.get_xticklabels(), rotation=50, fontsize = 20)
        plt.yticks(fontsize = 20)
        sns_boxplot.set_axisbelow(True)
        sns_boxplot.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
        sns_boxplot.set_title("Box plot of "+ featurey+ " by "+featurex+" in "+ name+" dataframe", fontsize=24)
        sns_boxplot.set_ylabel(featurey, fontsize=20)

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
            my_model, array_neuro_harmonized, s_data = harmonizationLearn(
                np.array(dataframe.drop(["FILE_ID", "SITE"], axis=1)),
                covars,
                return_s_data=True,
            )
            df_neuro_harmonized = pd.DataFrame(array_neuro_harmonized)
            df_neuro_harmonized.columns = dataframe.drop(
                ["FILE_ID", "SITE"], axis=1
            ).columns
            df_neuro_harmonized[
                [
                    "AGE_AT_SCAN",
                    "AGE_CLASS",
                    "DX_GROUP",
                    "SEX",
                    "FILE_ID",
                    "SITE",
                ]
            ] = dataframe[
                [
                    "AGE_AT_SCAN",
                    "AGE_CLASS",
                    "DX_GROUP",
                    "SEX",
                    "FILE_ID",
                    "SITE",
                ]
            ]
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
                    Dataframe containing neuroimaging data to correct with shape = (samples, features)
                    e.g. cortical thickness measurements, image voxels, etc
        confounder : string-like
                     Categorical feature to be accounted as confounder, which indicates batch (scanner) column name in covars
                     (e.g. "scanner") Default 'SITE_CLASS'.
        covariate : string-like
                    Contains the batch/scanner covariate as well as additional covariates (optional)
                    that should be preserved during harmonization. Default is 'AGE_AT_SCAN'.

        Returns
        -------

        dataframe_harmonized: dataframe-like
                              The dataframe containing harmonized data
        """
        array_combat_harmonized = neuroCombat(
            dat=dataframe.drop(["FILE_ID", "SITE"], axis=1).transpose(),
            covars=dataframe[[confounder, covariate]],
            batch_col=confounder,
        )["data"]
        df_combat_harmonized = pd.DataFrame(array_combat_harmonized.transpose())
        df_combat_harmonized.columns = dataframe.drop(
            ["FILE_ID", "SITE"], axis=1
        ).keys()
        df_combat_harmonized[
            ["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FILE_ID", "SITE"]
        ] = dataframe[
            ["AGE_AT_SCAN", "AGE_CLASS", "DX_GROUP", "SEX", "FILE_ID", "SITE"]
        ]
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
        if plot_heatmap is True:
            dataframe_restricted = dataframe[listoffeatures]
            heatmap = sns.heatmap(dataframe_restricted.corr(), annot=True)
            plt.show()
        listoffeatures = listoffeatures.drop(feature)
        X = dataframe[listoffeatures]
        y = dataframe[feature]
        return listoffeatures, X, y

    def remove_strings(self, dataframe):
        """
        Returns initial dataframe without columns containing strings.
        """
        name = self.retrieve_name(dataframe)
        if "FILE_ID" in dataframe.keys():
            dataframe = dataframe.drop(["FILE_ID"], axis=1)
        if "SITE" in dataframe.keys():
            dataframe = dataframe.drop(["SITE"], axis=1)
        type_dict = {
            col: dataframe[col].apply(lambda x: type(x)).unique().tolist()
            for col in dataframe.columns
        }
        value = [str]
        if value in type_dict.values():
            print(f"The dataframe '{name}' still contains '{value}'")
        else:
            print(f"The dataframe '{name}' is free of '{value}'")
        return dataframe

    def retrieve_name(self, var):
        """
        Retrieves name of variable.
        """
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        name = [var_name for var_name, var_val in callers_local_vars if var_val is var]
        name = str(name)
        name = name.replace('[','').replace(']','').replace('\'','')
        #print('Dataframe name: "{}"'.format(name))
        return name


if __name__ == "__main__":

    prep = Preprocessing()
    df = prep.read_file("data/FS_features_ABIDE_males.csv")
    Not_Normalized = prep(df, "not_normalized", plot_option=False)
    print(Not_Normalized.shape)
    Normalized = prep(df, "normalized", plot_option=False)
    print(Normalized.shape)
    Neuro_Harmonized = prep(df, "neuro_harmonized", plot_option=False)
    print(Neuro_Harmonized.shape)
    Combat_Harmonized = prep(df, "combat_harmonized", plot_option=False)
    print(Combat_Harmonized.shape)
    prep.plot_boxplot(Not_Normalized, "SITE", "TotalGrayVol")
    prep.plot_boxplot(Normalized, "SITE", "TotalGrayVol")
    prep.plot_boxplot(Neuro_Harmonized, "SITE", "TotalGrayVol")
    prep.plot_boxplot(Combat_Harmonized, "SITE", "TotalGrayVol")
    prep.remove_strings(Normalized)