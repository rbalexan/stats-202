import os

import pandas as pd


class Hw4(object):
  """Constructor"""
  def __init__(self, data_directory: Text='../data',
               figsize: Tuple[int, int]=(10, 10)):
    self.data_directory = data_directory
    self.figsize = figsize
    assert os.path.exists(self.data_directory), 'Directory not found.'

    # College data
    college_path = os.path.join(self.data_directory, 'College.csv')
    assert os.path.exists(college_path), 'College.csv not found.'
    logging.info("\tLoading %s", college_path)
    self.college = pd.read_csv(college_path, index_col='Unnamed: 0')
    self.college.columns = [x.replace('.', '_') for x in self.college.columns]


hw4 = Hw4()
