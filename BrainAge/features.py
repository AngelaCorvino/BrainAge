import pandas as pd

def file_reader(file_url):
  '''
  Read data features from url and return them in a dataframe
  '''
  dataset_file = file_url
  dataframe = pd.read_csv(dataset_file, sep=';')
  return dataframe
