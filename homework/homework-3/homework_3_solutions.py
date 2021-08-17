import os
from typing import List, Text, Tuple

from absl import logging
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn import svm
import statsmodels.api as sm
from statsmodels.formula.api import ols


class Hw3(object):
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

    # Auto data
    auto_path = os.path.join(self.data_directory, 'Auto.csv')
    assert os.path.exists(auto_path), 'Auto.csv not found.'
    logging.info('\tLoading %s', auto_path)
    self.auto = pd.read_csv(auto_path)
    self.auto.set_index("name", inplace=True)
    self.auto["origin"] = self.auto.origin.astype(str)

    # OJ data
    oj_path = os.path.join(self.data_directory, 'OJ.csv')
    assert os.path.exists(oj_path), 'OJ.csv not found.'
    logging.info('\tLoading %s', oj_path)
    self.oj = pd.read_csv(oj_path)
    self.oj['Store7'] = self.oj['Store7'].apply(lambda x: 1 if x=='Yes' else 0)

hw3 = Hw3()

# Chapter 6, Exercise 9 (p. 263). Don’t do parts (e), (f), and (g).
# (a)
np.random.seed(1)
training_idx = np.random.binomial(1, 0.8, hw3.college.shape[0])
data_train = hw3.college.loc[training_idx==1,]
data_test = hw3.college.loc[training_idx!=1,]

# (b)
formula = "Apps ~ " + ' + '.join(
    data_train.drop('Apps', axis=1).columns.to_list())
ols_fit = ols(formula, data=data_train).fit()
predictions = ols_fit.predict(data_test)
ols_mse = np.mean((data_test.Apps - predictions)**2)
print(ols_mse)

# (c)
# n.b. the penalty weight is taken directly off the minimized cv fits from the R implementation
ridge_fit = ols(formula, data=data_train).fit_regularized(L1_wt=0., alpha=362.9786)
predictions = ridge_fit.predict(data_test)
ridge_mse = np.mean((data_test.Apps - predictions)**2)
print(ridge_mse)

# (d)
# n.b. the penalty weight is taken directly off the minimized cv fits from the R implementation
lasso_fit = ols(formula, data=data_train).fit_regularized(L1_wt=1., alpha=0)
predictions = lasso_fit.predict(data_test)
lasso_mse = np.mean((data_test.Apps - predictions)**2)
print(lasso_mse)
print('Number of non-zero coef: %d' % (lasso_fit.params!=0.).sum())


## Chapter 7, Exercise 8 (p. 299).
# (a)
X = sm.add_constant(hw3.auto.weight)
y = hw3.auto.mpg
ols_fit = ols('mpg ~ weight', data=hw3.auto).fit()
new_x = np.linspace(hw3.auto.weight.min(), hw3.auto.weight.max(), 100)
pred1 = ols_fit.predict(sm.add_constant(new_x))

# Basis spline
transformed_X = dmatrix(
    "bs(weight, degree=3, knots=(2400, 3200, 4500), include_intercept=False)",
    {"weight": hw3.auto.weight}, return_type='dataframe')
columns = ['bs_' + str(i) for i in range(len(transformed_X.columns.to_list()))]
transformed_X.columns = columns
formula = 'mpg ~ ' + ' + '.join(transformed_X.columns.to_list())
transformed_X['mpg'] = hw3.auto.mpg
basis_spline_fit = ols(formula, data=transformed_X).fit()
transformed_xp = dmatrix(
    "bs(weight, degree=3, knots=(2400, 3200, 4500), include_intercept=False)",
    {"weight": new_x}, return_type='dataframe')
transformed_xp.columns = columns
pred2 = basis_spline_fit.predict(transformed_xp)
plt.scatter(hw3.auto.weight, y, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred1, color='g', label='Linear model')
plt.plot(xp, pred2, color='b', label='Basis spline')
plt.legend()
plt.savefig("./figures/python/ch7_ex8_part_a.png")

table = sm.stats.anova_lm(ols_fit, basis_spline_fit)
print(table)


## Chapter 9, Exercise 8 (p. 371).
# (a)
np.random.seed(1)
training_idx = np.random.binomial(1, 0.8, hw3.oj.shape[0])
data_train = hw3.oj.loc[training_idx==1]
data_test = hw3.oj.loc[training_idx!=1]

# (b)
clf = svm.SVC(C=0.01, kernel='linear')
clf.fit(data_train.drop('Purchase', axis=1), data_train.Purchase)
clf

# (c)
train_pred = clf.predict(data_train.drop('Purchase', axis=1))
error_linear_train = np.mean(train_pred != data_train.Purchase)
test_pred = clf.predict(data_test.drop('Purchase', axis=1))
error_linear_test = np.mean(test_pred != data_test.Purchase)
print('Train error: %0.3f' % error_linear_train)
print('Test error: %0.3f' % error_linear_test)

# (d)
# n.b. The optimal cost was directly taken from the R implementation
clf = svm.SVC(C=7.079458, kernel='linear')
clf.fit(data_train.drop('Purchase', axis=1), data_train.Purchase)
clf

# (e)
train_pred = clf.predict(data_train.drop('Purchase', axis=1))
error_linear_train_tune = np.mean(train_pred != data_train.Purchase)
test_pred = clf.predict(data_test.drop('Purchase', axis=1))
error_linear_test_tune = np.mean(test_pred != data_test.Purchase)
print('Train error: %0.3f' % error_linear_train_tune)
print('Test error: %0.3f' % error_linear_test_tune)

# (f)
# n.b. The optimal cost was directly taken from the R implementation
clf = svm.SVC(C=0.6309573, kernel='rbf')
clf.fit(data_train.drop('Purchase', axis=1), data_train.Purchase)
clf
train_pred = clf.predict(data_train.drop('Purchase', axis=1))
error_linear_train_tune = np.mean(train_pred != data_train.Purchase)
test_pred = clf.predict(data_test.drop('Purchase', axis=1))
error_linear_test_tune = np.mean(test_pred != data_test.Purchase)
print('Train error: %0.3f' % error_linear_train_tune)
print('Test error: %0.3f' % error_linear_test_tune)

# (g)
# n.b. The optimal cost was directly taken from the R implementation
clf = svm.SVC(C=7.079458, kernel='poly', degree=2)
clf.fit(data_train.drop('Purchase', axis=1), data_train.Purchase)
clf
train_pred = clf.predict(data_train.drop('Purchase', axis=1))
error_linear_train = np.mean(train_pred != data_train.Purchase)
test_pred = clf.predict(data_test.drop('Purchase', axis=1))
error_linear_test = np.mean(test_pred != data_test.Purchase)
print('Train error: %0.3f' % error_linear_train)
print('Test error: %0.3f' % error_linear_test)

