import unittest
import os
import sys

#to run locally
package_name = "../BrainAge"
#package_name = "BrainAge"

sys.path.insert(0, package_name)

from features import Preprocessing

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
        prep = Preprocessing()
        dataframe = prep.file_reader(self.data)
        assert dataframe.size == 387960
        assert dataframe.shape == (915, 424)

    def test_add_features(self):
        prep = Preprocessing()
        dataframe = prep.file_reader(self.data)
        prep.add_features(dataframe)
        assert 'Site' in dataframe.keys()
        assert 'TotalWhiteVol' in dataframe.keys()
        
    def test_create_binning(self):
        prep = Preprocessing()
        dataframe = prep.file_reader(self.data)
        prep.create_binning(dataframe)
        assert dataframe.shape == (915, 425)

    def test_file_split(self):
        prep = Preprocessing()
        dataframe = prep.file_reader(self.data)
        df_AS, df_TD = prep.file_split(dataframe)
        assert df_AS.shape == (451, 424)
        assert df_TD.shape == (464, 424)

    def test_feature_selection(self):
        prep = Preprocessing()
        dataframe = prep.file_reader(self.data)
        features, _, _ = prep.feature_selection(dataframe, 'AGE_AT_SCAN', False)
        assert features.shape == (13, )
        _, df_TD = prep.file_split(dataframe)
        features_TD, _, _ = prep.feature_selection(df_TD, 'AGE_AT_SCAN', False)
        assert features_TD.shape == (16, )

if __name__ == "__main__":
    unittest.main()
