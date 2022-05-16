import unittest
import os
import sys

package_name = "BrainAge"
# package_root = os.path.abspath("..")
# sys.path.insert(0, package_root)
# sys.path.insert(0, os.path.join(package_root, package_name))

sys.path.insert(0,package_name)

from features import Utilities
from regression import Regression


class TestBrainAge(unittest.TestCase):
    """
    Class for testing our code
    """


    def test_file_reader(self):
        util = Utilities(
            "data/FS_features_ABIDE_males.csv"
        )
        dataframe = util.file_reader()
        df_AS, df_TD = util.file_split()
        assert dataframe.size == 388875
        assert dataframe.shape == (915, 425)
        assert df_AS.shape == (451, 425)
        assert df_TD.shape == (464, 425)


    def test_file_split(self):
        util = Utilities(
            "data/FS_features_ABIDE_males.csv"
        )
        df_AS, df_TD = util.file_split()
        assert df_AS.shape == (451, 425)
        assert df_TD.shape == (464, 425)

    def test_feature_selection(self):
        util = Regression(
            "/Users/angelacorvino/Documents/GitHub/BrainAge/data/FS_features_ABIDE_males.csv"
        )
        features = util.feature_selection(heatmap=False)
        assert features.shape == (16, )


if __name__ == "__main__":
    unittest.main()
