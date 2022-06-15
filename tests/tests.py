import unittest
import os
import sys

#to run locally
#package_name = "../BrainAge"
package_name = "BrainAge"

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

    def test_read_file(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        self.assertEqual(dataframe.size, 387960, 'Wrong Size')
        self.assertEqual(dataframe.shape, (915, 424), 'Wrong Shape')

    def test_add_features(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        prep.add_features(dataframe)
        self.assertIn('SITE', dataframe.keys(), 'SITE was not added')
        self.assertIn('TotalWhiteVol', dataframe.keys(), 'TotalWhiteVol was not added')
        self.assertEqual(dataframe.shape, (915, 426), 'Two features were not added')
        
    def test_add_age_binning(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        prep.add_age_binning(dataframe)
        self.assertEqual(dataframe.shape, (915, 425), 'Wrong Shape')
        self.assertIn('AGE_CLASS', dataframe.keys(), 'AGE_CLASS was not added')
        
    def test_add_site_binning(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        prep.add_features(dataframe)
        dataframe = prep.add_site_binning(dataframe)
        self.assertIn('SITE_CLASS', dataframe.keys(), 'SITE_CLASS was not added')
        self.assertEqual(dataframe.shape, (915, 427), 'SITE_CLASS was not added')

    def test_split_file(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        df_AS, df_TD = prep.split_file(dataframe)
        assert df_AS.shape == (451, 424)
        assert df_TD.shape == (464, 424)

    def test_feature_selection(self):
        prep = Preprocessing()
        dataframe = prep.read_file(self.data)
        features, _, _ = prep.feature_selection(dataframe, 'AGE_AT_SCAN', False)
        self.assertEqual(features.shape, (13, ), 'Wrong number of selected features')

if __name__ == "__main__":
    unittest.main()
