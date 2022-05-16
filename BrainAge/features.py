import pandas as pd


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
        dataset_file = self.file_url
        df = pd.read_csv(dataset_file, sep=";")
<<<<<<< HEAD
        df['Site'] =df.FILE_ID.apply(lambda x: x.split('_')[0])
        return df


=======
        df["Site"] = df.FILE_ID.apply(lambda x: x.split("_")[0])
        return df

>>>>>>> angela
    def file_split(self):
        """
        Split dataframe in healthy(control) and autistic subjects
        """
        df_AS = self.df.loc[self.df.DX_GROUP == 1]
        df_TD = self.df.loc[self.df.DX_GROUP == -1]
        return df_AS, df_TD
