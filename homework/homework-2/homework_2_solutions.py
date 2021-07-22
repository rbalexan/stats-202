import os
from typing import List, Text, Tuple

from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from resample.bootstrap import bootstrap
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm

class Hw2(object):
  """Constructor"""
  def __init__(self, data_directory: Text='../data',
               figsize: Tuple[int, int]=(10, 10)):
    self.data_directory = data_directory
    self.figsize = figsize
    assert os.path.exists(self.data_directory), 'Directory not found.'

    # Weekly data
    weekly_path = os.path.join(self.data_directory, 'Weekly.csv')
    assert os.path.exists(weekly_path), 'Weekly.csv not found.'
    logging.info("\tLoading %s", weekly_path)
    self.weekly = pd.read_csv(weekly_path)

    # Default data
    default_path = os.path.join(self.data_directory, 'Default.csv')
    assert os.path.exists(default_path), 'Default.csv not found.'
    logging.info("\tLoading %s", default_path)
    self.default = pd.read_csv(default_path)
    self.default.default = self.default.default.apply(
      lambda x: 1 if x=="Yes" else 0)

    # Simulation data
    sim_path = os.path.join(self.data_directory, 'ch5_ex8.csv')
    assert os.path.exists(sim_path), 'ch5_ex8.csv not found.'
    logging.info("\tLoading %s", sim_path)
    self.sim_data = pd.read_csv(sim_path)

    # Boston data
    boston_path = os.path.join(self.data_directory, 'Boston.csv')
    assert os.path.exists(boston_path), 'Boston.csv not found.'
    logging.info("\tLoading %s", boston_path)
    self.boston = pd.read_csv(boston_path)


hw2 = Hw2()

## Chapter 4, Exercise 10 (p. 171).
# (a)
corr_matrix = hw2.weekly.corr()
print(corr_matrix)

fig = scatter_matrix(hw2.weekly, figsize=hw2.figsize)
plt.savefig("./figures/python/ch4_ex10_part_a.png")

# (b)
features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
X = sm.add_constant(hw2.weekly[features])
y = hw2.weekly.Direction.apply(lambda x: 1 if x == "Up" else 0)
logit_fit = sm.Logit(y, X).fit()
print(logit_fit.summary())

# (c)
features = ['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']
X = sm.add_constant(hw2.weekly[features])
y = hw2.weekly.Direction.apply(lambda x: 1 if x == "Up" else 0)
logit_fit = sm.Logit(y, X).fit()
prob_y = logit_fit.predict(X)
pred_y = pd.Series(["Up" if x >= 0.5 else "Down" for x in prob_y])
true_y = hw2.weekly.Direction
print(pd.crosstab(pred_y, true_y))

# (d)
train_idx = (hw2.weekly.Year <= 2008)
data_train = hw2.weekly.loc[train_idx].reset_index()
data_test = hw2.weekly.loc[~train_idx].reset_index()
features = ["Lag2"]
X = sm.add_constant(data_train[features])
y = data_train.Direction.apply(lambda x: 1 if x == "Up" else 0)
logit_fit = sm.Logit(y, X).fit()
X_test = sm.add_constant(data_test[features])
prob_y = logit_fit.predict(X_test)
pred_y = pd.Series(["Up" if x >= 0.5 else "Down" for x in prob_y])
print(pd.crosstab(pred_y, true_y))

# (e)
X = data_train[features].values
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
pred_y = pd.Series(clf.predict(data_test[features]))
print(pd.crosstab(pred_y, true_y))

# (f)
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)
pred_y = pd.Series(clf.predict(data_test[features]))
print(pd.crosstab(pred_y, true_y))

# (g)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
pred_y = pd.Series(neigh.predict(data_test[features]))
print(pd.crosstab(pred_y, true_y))

## Chapter 5, Exercise 2 (p. 197).
# (g)
def prob_not_j(n: int):
  return (1 - 1/float(n))**n

_, ax = plt.subplots(figsize=hw2.figsize)
ax.set_xlabel("n")
ax.set_ylabel("Probability")
ax.plot([prob_not_j(x+1) for x in range(100000)], linestyle = 'solid')
plt.savefig("./figures/python/ch5_ex2_part_g.png")

# (h)
store = []
for i in range(10000):
  store.append(np.sum(np.random.choice(100, size=100, replace=True)==4) > 0)

print(np.mean(store))


## Chapter 5, Exercise 5 (p. 198).
# (a)
features = ["income", "balance"]
X = sm.add_constant(hw2.default[features])
y = hw2.default.default
logit_fit = sm.Logit(y, X).fit()
print(logit_fit.summary())

# (b)
def q_5_ex_5_part_b(features:List[Text] = ["income", "balance"], seed:int = 1
                   ) -> float:
  X = sm.add_constant(hw2.default[features])
  y = hw2.default.default
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, random_state=seed)
  logit_fit = sm.Logit(y_train, X_train).fit()
  prob_y = logit_fit.predict(X_test)
  pred_y = [1 if x >= 0.5 else 0 for x in prob_y]
  validation_error = np.mean([x!=y for x, y in zip(pred_y, y_test)])
  return validation_error

print(q_5_ex_5_part_b())

# (c)
valid_set_err = []
for seed in range(3):
  err = q_5_ex_5_part_b(seed=seed)
  valid_set_err.append(err)

print(valid_set_err)

# (d)
hw2.default['student_bi'] = hw2.default.student.apply(
  lambda x: 1 if x=="Yes" else 0)
err = q_5_ex_5_part_b(features=["income", "balance", "student_bi"])
print(err)


## Chapter 5, Exercise 6 (p. 199).
# (a)
features = ["income", "balance"]
X = sm.add_constant(hw2.default[features])
y = hw2.default.default
logit_fit = sm.Logit(y, X).fit()
print(logit_fit.summary())

# (b)
def boot_fn(data_frame:pd.DataFrame) -> List[float]:
  X = sm.add_constant(data_frame[:, :-1])
  y = data_frame[:,-1]
  logit_fit = sm.Logit(y, X).fit(disp=False)
  coef = logit_fit.params.tolist()
  return coef

# (c)
variables = ["income", "balance", "default"]
boot_coef = bootstrap(a=hw2.default[variables], f=boot_fn, b=1000)
boot_stderr = boot_coef.std(axis=0)
print(boot_stderr)


## Chapter 5, Exercise 8 (p. 200).
# (b)
_, ax = plt.subplots(figsize=hw2.figsize)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.scatter(hw2.sim_data.x, hw2.sim_data.y)
plt.savefig("./figures/python/ch5_ex8_part_b.png")

# (c)
def ch_5_ex_8_pt_c1(seed=1):
  X = sm.add_constant(hw2.sim_data.x.values)
  y = hw2.sim_data.y.values
  loo = KFold(n_splits=hw2.sim_data.shape[0], random_state=seed)
  loocv_errors = []
  for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ols_fit = sm.OLS(y_train, X_train).fit()
    pred_y = ols_fit.predict(X_test)
    loocv_errors.append((pred_y - y_test)**2)

  loocv_error = np.mean(loocv_errors)
  return loocv_error

print(ch_5_ex_8_pt_c1())

def ch_5_ex_8_pt_c2(seed=1):
  sim_data = hw2.sim_data.copy()
  sim_data['x2'] = sim_data.x**2
  X = sm.add_constant(sim_data[['x', 'x2']].values)
  y = sim_data.y.values
  loo = KFold(n_splits=sim_data.shape[0], random_state=seed)
  loocv_errors = []
  for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ols_fit = sm.OLS(y_train, X_train).fit()
    pred_y = ols_fit.predict(X_test)
    loocv_errors.append((pred_y - y_test)**2)

  loocv_error = np.mean(loocv_errors)
  return loocv_error

print(ch_5_ex_8_pt_c2())

def ch_5_ex_8_pt_c3(seed=1):
  sim_data = hw2.sim_data.copy()
  sim_data['x2'] = sim_data.x**2
  sim_data['x3'] = sim_data.x**3
  X = sm.add_constant(sim_data[['x', 'x2', 'x3']].values)
  y = sim_data.y.values
  loo = KFold(n_splits=sim_data.shape[0], random_state=seed)
  loocv_errors = []
  for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ols_fit = sm.OLS(y_train, X_train).fit()
    pred_y = ols_fit.predict(X_test)
    loocv_errors.append((pred_y - y_test)**2)

  loocv_error = np.mean(loocv_errors)
  return loocv_error

print(ch_5_ex_8_pt_c2())

def ch_5_ex_8_pt_c4(seed=1):
  sim_data = hw2.sim_data.copy()
  sim_data['x2'] = sim_data.x**2
  sim_data['x3'] = sim_data.x**3
  sim_data['x4'] = sim_data.x**4
  X = sm.add_constant(sim_data[['x', 'x2', 'x3', 'x4']].values)
  y = sim_data.y.values
  loo = KFold(n_splits=sim_data.shape[0], random_state=seed)
  loocv_errors = []
  for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ols_fit = sm.OLS(y_train, X_train).fit()
    pred_y = ols_fit.predict(X_test)
    loocv_errors.append((pred_y - y_test)**2)

  loocv_error = np.mean(loocv_errors)
  return loocv_error

print(ch_5_ex_8_pt_c4())

# (d)
loocv_errors = [ch_5_ex_8_pt_c1(seed=2), ch_5_ex_8_pt_c2(seed=2),
                ch_5_ex_8_pt_c3(seed=3), ch_5_ex_8_pt_c4(seed=4)]
print(loocv_errors)

# (f)
X = sm.add_constant(hw2.sim_data.x)
y = hw2.sim_data.y
ols_fit = sm.OLS(y, X).fit()
print(ols_fit.summary())

sim_data = hw2.sim_data.copy()
sim_data['x2'] = sim_data.x**2
X = sm.add_constant(sim_data[["x", "x2"]])
y = sim_data.y
ols_fit = sm.OLS(y, X).fit()
print(ols_fit.summary())

sim_data = hw2.sim_data.copy()
sim_data['x2'] = sim_data.x**2
sim_data['x3'] = sim_data.x**3
X = sm.add_constant(sim_data[["x", "x2", "x3"]])
y = sim_data.y
ols_fit = sm.OLS(y, X).fit()
print(ols_fit.summary())

sim_data = hw2.sim_data.copy()
sim_data['x2'] = sim_data.x**2
sim_data['x3'] = sim_data.x**3
sim_data['x4'] = sim_data.x**4
X = sm.add_constant(sim_data[["x", "x2", "x3", "x4"]])
y = sim_data.y
ols_fit = sm.OLS(y, X).fit()
print(ols_fit.summary())


## Chapter 5, Exercise 9 (p. 201).
# (a)
estimated_mean = np.mean(hw2.boston.medv)
print(estimated_mean)

# (b)
standard_error = np.sqrt(np.var(hw2.boston.medv) / hw2.boston.shape[0])
print(standard_error)

# (c)
boot_mean = bootstrap(a=hw2.boston.medv.values, f=np.mean, b=1000,
                      random_state=1)
boot_stderr = boot_mean.std(axis=0)
print(boot_stderr)

# (d)
crit_value = stats.norm.ppf(.975)
ci_boot = [estimated_mean + x * crit_value * boot_stderr for x in [-1, 1]]
print(ci_boot)

unbiased_variance = np.var(hw2.boston.medv, ddof=1)
standard_error = np.sqrt(unbiased_variance / hw2.boston.shape[0])
crit_value = stats.t.ppf(.975, hw2.boston.shape[0] - 1)
ci_t = [estimated_mean + x * crit_value * standard_error for x in [-1, 1]]
print(ci_t)

# (e)
estimated_median = np.median(hw2.boston.medv)
print(estimated_median)

# (f)
boot_median = bootstrap(a=hw2.boston.medv.values, f=np.median, b=1000,
                        random_state=1)
boot_stderr = boot_median.std(axis=0)
print(boot_stderr)

# (g)
q10_estimate = np.percentile(hw2.boston.medv, 10)
print(q10_estimate)

# (h)
def tenth_percentile(x):
  return np.percentile(x, 10)

boot_tenth = bootstrap(a=hw2.boston.medv.values, f=tenth_percentile,
                       b=1000, random_state=1)
boot_stderr = boot_tenth.std(axis=0)
print(boot_stderr)
