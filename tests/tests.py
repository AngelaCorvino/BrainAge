import unittest
import os
import sys
package_name = 'BrainAge'
package_root = os.path.abspath('..')
sys.path.insert(0, package_root)
sys.path.insert(0, os.path.join(package_root, package_name))

from features import Utilities
from regression import Regression

class TestBrainAge(unittest.TestCase):
	'''
	Class for testing our code
	'''
	def __init__(self):
		
		self.util=self.test_file_reader()

	def test_file_reader(self):
		util = Utilities('/data/FS_features_ABIDE_males.csv')
		dataframe = util.file_reader()
		df_AS,df_TD=util.file_split()
		assert dataframe.size == 388875
		assert dataframe.shape == (915, 425)
		assert df_AS.shape == (451, 425)
		assert df_TD.shape == (464, 425)
		return util


	def test_file_split(self):
		df_AS,df_TD=self.util.file_split()
		assert df_AS.shape == (451, 425)
		assert df_TD.shape == (464, 425)

	def test_feature_selection(self):
		util=Regression('/data/FS_features_ABIDE_males.csv')
		features=util.feature_selection()
		assert features.shape == (17,)




if __name__=='__main__':
	unittest.main()
