import unittest
import os
import sys

#to run locally
#
package_name = "../BrainAge" 
#package_name = "BrainAge"


sys.path.insert(0, package_name)

from features import Utilities
from regression import Regression


class TestBrainAge(unittest.TestCase):
    """
    Class for testing our code
    """
    def setUp(self):
        """
        Initialize the class.
        """
        self.data = package_name + "/data/FS_features_ABIDE_males.csv"
        
    def test_file_reader(self):
        util = Utilities(self.data)
        dataframe = util.file_reader()
        df_AS, df_TD = util.file_split()
        assert dataframe.size == 387960
        assert dataframe.shape == (915, 424)
        assert df_AS.shape == (451, 426)
        assert df_TD.shape == (464, 426)
        
    def test_add_feature(self):
        util = Utilities(self.data)
        dataframe = util.file_reader()
        dataframe = util.add_features()
        assert 'Site' in dataframe.keys()
        assert 'TotalWhiteVol' in dataframe.keys()

    def test_file_split(self):
        util = Utilities(self.data)
        df_AS, df_TD = util.file_split()
        assert df_AS.shape == (451, 426)
        assert df_TD.shape == (464, 426)

    def test_feature_selection(self):
        util = Utilities(self.data)
        features = util.feature_selection('AGE_AT_SCAN', False)
        assert features.shape == (16, )

if __name__ == "__main__":
    unittest.main()
