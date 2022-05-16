import pandas as pd

class Utilities():
	'''
	Basic utilities functions for data
	'''

	def __init__(self,file_url):
		'''
		Initialize the class.
		'''
		self.file_url = file_url
		self.df = file_reader(self.file_url)

	def file_reader(self):
		'''
		Read data features from url and return them in a dataframe
		'''
		dataset_file = self.file_url
		df = pd.read_csv(dataset_file, sep=';')
		return df
