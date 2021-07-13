import collections
import os
from typing import Text, Tuple

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import cluster
from scipy.stats import pearsonr
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence


class Hw1(object):
  """Constructor"""
  def __init__(self, data_directory: Text='../data',
               figsize: Tuple[int, int]=(10, 10)):
    self.data_directory = data_directory
    self.figsize = figsize
    assert os.path.exists(self.data_directory), 'Directory not found.'

    # USArrests data
    us_arrests_path = os.path.join(self.data_directory, 'USArrests.csv')
    assert os.path.exists(us_arrests_path), 'USArrests.csv not found.'
    logging.info('\tLoading %s', us_arrests_path)
    self.us_arrests = pd.read_csv(us_arrests_path, index_col='Unnamed: 0')

    # Auto data
    auto_path = os.path.join(self.data_directory, 'Auto.csv')
    assert os.path.exists(auto_path), 'Auto.csv not found.'
    logging.info('\tLoading %s', auto_path)
    self.auto = pd.read_csv(auto_path)
    self.auto.set_index("name", inplace=True)
    self.auto["origin"] = self.auto.origin.astype(str)

    # Simulation data
    sim_path = os.path.join(self.data_directory, 'ch3_q14.csv')
    assert os.path.exists(sim_path), 'ch3_q14.csv not found.'
    logging.info("\tLoading %s", sim_path)
    self.sim_data = pd.read_csv(sim_path)


hw1 = Hw1()

## Chapter 10, Exercise 9 (p. 416). ##
# (a)
clusters = cluster.hierarchy.linkage(hw1.us_arrests, method='complete')
_, ax = plt.subplots(figsize=hw1.figsize)
dendo = cluster.hierarchy.dendrogram(clusters,
                                     orientation='top',
                                     labels=hw1.us_arrests.index.to_list(),
                                     distance_sort='descending',
                                     show_leaf_counts=True,
                                     ax=ax)
plt.savefig("./figures/python/ch10_ex9_part_a.png")

# (b)
clusts = cluster.hierarchy.cut_tree(clusters, n_clusters=[3])
groups = collections.defaultdict(list)
for state, group in zip(hw1.us_arrests.index.to_list(), clusts[:,0]):
  groups[group].append(state)

for group, states in groups.items():
  print("Group " + str(group) + ":")
  print(", ".join(states), '\n')

# (c)
us_arrests_std = pd.DataFrame(preprocessing.scale(hw1.us_arrests, axis=0),
                              columns=hw1.us_arrests.columns)
clusters = cluster.hierarchy.linkage(us_arrests_std, method='complete')
_, ax = plt.subplots(figsize=hw1.figsize)
dendo = cluster.hierarchy.dendrogram(clusters,
                                     orientation='top',
                                     labels=hw1.us_arrests.index.to_list(),
                                     distance_sort='descending',
                                     show_leaf_counts=True,
                                     ax=ax)
plt.savefig("./figures/python/ch10_ex9_part_c.png")


## Chapter 3, Exercise 9 (p. 122). ##
# (a)
fig = scatter_matrix(hw1.auto, figsize=hw1.figsize)
plt.savefig("./figures/python/ch3_ex9_part_a.png")

# (b)
corr_matrix = hw1.auto.corr()
print(corr_matrix)

# (c)
formula = ("mpg ~ cylinders + displacement + horsepower + weight "
           "+ acceleration + year + origin")
ols_fit = ols(formula, data=hw1.auto).fit()
print(ols_fit.summary())

# (d)
_, ax = plt.subplots(figsize=hw1.figsize)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
figure = ax.scatter(ols_fit.fittedvalues, ols_fit.resid)
plt.savefig("./figures/python/ch3_ex9_part_d1.png")

outliers_influence = OLSInfluence(ols_fit)
figure = sm.qqplot(outliers_influence.get_resid_studentized_external())
plt.savefig("./figures/python/ch3_ex9_part_d2.png")

outliers_influence = OLSInfluence(ols_fit)
_, ax = plt.subplots(figsize=hw1.figsize)
ax.set_xlabel("Fitted values")
ax.set_ylabel("sqrt(studentized residuals)")
figure = ax.scatter(ols_fit.fittedvalues,
                    outliers_influence.get_resid_studentized_external().apply(
                      lambda x: np.sqrt(abs(x))))
plt.savefig("./figures/python/ch3_ex9_part_d3.png")

_, ax = plt.subplots(figsize=hw1.figsize)
figure = sm.graphics.influence_plot(ols_fit, ax=ax)
plt.savefig("./figures/python/ch3_ex9_part_d4.png")

# (e)
formula = ("mpg ~ cylinders + displacement + horsepower + weight "
           "+ acceleration*year*origin")
ols_fit2 = ols(formula, data=hw1.auto).fit()
print(ols_fit2.summary())

# (f)
hw1.auto["displacement_sq"] = hw1.auto.displacement**2
hw1.auto["log_year"] = hw1.auto.year.apply(np.log)
formula = ("mpg ~ cylinders + displacement + horsepower + weight "
           "+ acceleration + year + origin + displacement_sq + log_year")
ols_fit3 = ols(formula, data=hw1.auto).fit()
print(ols_fit3.summary())

## Chapter 3, Exercise 14 (p. 125). ##
# (b)
corr, _ = pearsonr(hw1.sim_data.x1, hw1.sim_data.x2)
print(corr)

_, ax = plt.subplots(figsize=hw1.figsize)
ax.set_xlabel("x2")
ax.set_ylabel("x1")
figure = ax.scatter(hw1.sim_data.x2, hw1.sim_data.x1)
plt.savefig("./figures/python/ch3_ex14_part_b.png")

# (c)
formula = ("y ~ x1 + x2")
ols_fit = ols(formula, data=hw1.sim_data).fit()
ols_summary = ols_fit.summary()
print(ols_summary)

# (d)
formula = ("y ~ x1")
ols_fit = ols(formula, data=hw1.sim_data).fit()
ols_summary = ols_fit.summary()

# (e)
formula = ("y ~ x2")
ols_fit = ols(formula, data=hw1.sim_data).fit()
ols_summary = ols_fit.summary()

# (g)
extra_obs = pd.DataFrame({"x1": [0.1], "x2":[0.8], "y": [6]})
updated_sim_data = hw1.sim_data.append(extra_obs)
formula = ("y ~ x1 + x2")
ols_fit = ols(formula, data=updated_sim_data).fit()
print(ols_fit.summary())

extra_obs = pd.DataFrame({"x1": [0.1], "x2":[0.8], "y": [6]})
updated_sim_data = hw1.sim_data.append(extra_obs)
formula = ("y ~ x1")
ols_fit = ols(formula, data=updated_sim_data).fit()
print(ols_fit.summary())

extra_obs = pd.DataFrame({"x1": [0.1], "x2":[0.8], "y": [6]})
updated_sim_data = hw1.sim_data.append(extra_obs)
formula = ("y ~ x2")
ols_fit = ols(formula, data=updated_sim_data).fit()
print(ols_fit.summary())
