import pandas as pd

class Utilities():
	'''
	Basic utilities functions for data
	'''
	
	def __init__(self, df):
        """
        Initialize the class.
        """
        self.df= df
        
	def file_reader(self, file_url):
  	'''
 	Read data features from url and return them in a dataframe
  	'''
  	dataset_file = file_url
  	self.df = pd.read_csv(dataset_file, sep=';')
  	return self.df
