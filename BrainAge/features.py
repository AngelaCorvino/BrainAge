import pandas as pd
import matplotlib.pyplot as plt

class Utilities:
    """
    Basic utilities functions for data
    """
    def __init__(self, file_url):
        """
        Initialize the class.
        """
        self.file_url = file_url
        self.df = self.file_reader()
        (self.df_AS, self.df_TD) = self.file_split()

    def file_reader(self):
        """
        Read data features from url and return them in a dataframe
        """
        df = pd.read_csv(self.file_url, sep = ";")
        return df
        
    def add_features(self):
        """
        Add columns with derived features
        """
        self.df['TotalWhiteVol'] = self.df.lhCerebralWhiteMatterVol+self.df.rhCerebralWhiteMatterVol
        self.df['Site'] = self.df.FILE_ID.apply(lambda x: x.split('_')[0])
        return

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
            self.df_ASD[self.gdf_ASD.loc[:, feature]>0].hist([feature])
        plt.show()
        return
        
if __name__ == "__main__":
    util = Utilities("data/FS_features_ABIDE_males.csv")
    util.add_features()
    util.file_split()
    util.plot_histogram('AGE_AT_SCAN')
