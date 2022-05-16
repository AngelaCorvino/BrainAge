from features import Utilities

class regression():
    """
    Class describing regression model.
    Parameters
    ----------
    file_url : string-like
        The path that point to the data.

    """

    def __init__(self, file_url):
        """
        Constructor.
        """
        self.file_url=file_url
        self.util=Utilities(file_url)
        (self.df_AS,self.df_TD) =self.util.file_split()

    def feature_selection(self):
        print('ciao')
        return
