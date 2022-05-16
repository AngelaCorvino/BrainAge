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
		assert dataframe.size == 387960
		assert dataframe.shape == (915, 424)
if __name__=='__main__':
	unittest.main()
