import unittest
import sys
sys.path.insert(0, 'BrainAge')
#import os
#import numpy as np

from features import Utilities

class TestBrainAge(unittest.TestCase):
	'''
	Class for testing our code
	'''
	def test_file_reader(self):
		util = Utilities('https://raw.githubusercontent.com/AngelaCorvino/BrainAge/main/BrainAge/data/FS_features_ABIDE_males.csv')
		dataframe = util.file_reader()
		df_AS,df_TD=util.file_split()
		assert dataframe.size == 388875
		assert dataframe.shape == (915, 425)
		assert df_AS.shape == (451, 425)
		assert df_TD.shape == (464, 425)

	def test_file_split(self):
		util = Utilities('https://raw.githubusercontent.com/AngelaCorvino/BrainAge/main/BrainAge/data/FS_features_ABIDE_males.csv')
		df_AS,df_TD=util.file_split()
		assert df_AS.shape == (451, 425)
		assert df_TD.shape == (464, 425)


if __name__=='__main__':
	unittest.main()
