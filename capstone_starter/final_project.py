#   Outline:
#
#   1. Linear regression between age and income
#   2. Multiple linear regression to see if we can get more accuracy
#   3. Classification method(s) to see if we can determine income bracket
#

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import random
import timeit


# get rid of warnings I can do nothing about
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Create your df here:
df = pd.read_csv('profiles.csv')

#print('Rows beginning: ' , len(df))
#take columns, then get rid of NAN
dataset = df.loc[:,['age', 'education', 'income', 'drinks', 'smokes', 'drugs']]
dataset.dropna(inplace=True)

print('Rows left: ' , len(dataset))

print(dataset.income.value_counts())
#print(dataset.age.value_counts())
#print(dataset.drinks.value_counts())
#print(dataset.smokes.value_counts())
#print(dataset.drugs.value_counts())

##  We have lots of missing data in the income column - too many to throw away.
##  Need to deal with them somehow.  Let's replace with the median values after
##  getting rid of the outliers.

#get rid of the "millionaires" and "half-millionaire" outliers
dataset = dataset[dataset.income != 1000000]
dataset = dataset[dataset.income != 500000]

imputer = Imputer(missing_values=-1, strategy='median', axis=0)
dataset[['income']] = imputer.fit_transform(dataset[['income']])

#print(dataset.income.value_counts())

################################################################
##  AGE VS. INCOME PLOT

train, test = train_test_split(dataset, test_size = 0.2)

Xtr = train[['age']]
ytr = train[['income']]

start = timeit.default_timer()
regr = LinearRegression()
regr.fit(Xtr,ytr)

X = test[['age']]
y = test[['income']]
y_predicted = regr.predict(X)
stop = timeit.default_timer()

print('Time to run simple linear regression: ', end = '')
print(stop - start)

print('Coefficients', regr.coef_)
print('Age v. Education Regression Train Score: ', regr.score(Xtr,ytr))
print('Age v. Education Regression Test Score: ', regr.score(X,y))

plt.xlabel('Age')
plt.ylabel('Income')
plt.scatter(X,y,alpha=0.01)
plt.plot(X, y_predicted)
plt.xlim(16,80)
plt.show()

start = timeit.default_timer()
################################################################
##  CREATE NEW DATA COLUMN FOR ESTIMATED EDUCATION YEARS

#this is highly arbitrary and unscientific, but convert responses
#to estimated years of schooling

#drop the space camp yokels
dataset = dataset[~dataset.education.str.contains('space')]

#assign potential responses to groups
sub_high_school = ['dropped out of high school',
                   'working on high school'] #10
high_school = ['high school',
               'working on two-year college'                #12
               'dropped out of two-year college',
               'graduated from high school',
               'working on two-year college']
some_college = ['two-year college',
                'graduated from two-year college',
                'dropped out of college/university',
                'working on college/university']    #14
college_grad = ['graduated from college/university',
                'working on masters program',
                'college/university',
                'working on law school',
                'working on med school',
                'dropped out of masters program',
                'dropped out of ph.d program',
                'masters program',
                'ph.d program',
                'law school',
                'dropped out of law school',
                'dropped out of med school',
                'med school',
                'dropped out of two-year college']  #16
masters = ['graduated from masters program',
           'working on ph.d program']    #17
postgraduate = ['graduated from ph.d program',
                'graduated from law school',
                'graduated from med school']    #19

def set_ed_years (row):    
    if row['education'] in sub_high_school:
        return 10
    if row['education'] in high_school:
        return 12
    if row['education'] in some_college:
        return 14
    if row['education'] in college_grad:
        return 16
    if row['education'] in masters:
        return 17
    if row['education'] in postgraduate:
        return 19
    return 0

dataset['education_years'] = dataset.apply (lambda row: set_ed_years(row), axis=1)
stop = timeit.default_timer()

print('Time to categorize education years: ', end = '')
print(stop - start)

#print(dataset['education_years'].head())

################################################################
##  CREATE NEW DATA COLUMNS FOR DRINKS, SMOKES and DRUGS

def drink_code_me (row):
    if row['drinks'] == 'not at all':
        return 0
    if row['drinks'] == 'rarely':
        return 1
    if row['drinks'] =='socially':
        return 2
    if row['drinks'] == 'often':
        return 3
    if row['drinks'] == 'very often':
        return 4
    if row['drinks'] == 'desperately':
        return 5
    return -1

dataset['drinks_code'] = dataset.apply (lambda row: drink_code_me(row), axis=1)

#drinks_mapping = {'not at all':0, 'rarely':1, 'socially':2, 'often':3,
#                  'very often':4, 'desperately':5}
#dataset['drinks_code'] = dataset.drinks.map(drinks_mapping)
print('drinks_code mapped.')

smokes_mapping = {'no':0, 'sometimes':1, 'when drinking':2, 'yes':3, 'trying to quit':3}
dataset['smokes_code'] = dataset.smokes.map(smokes_mapping)
print('smokes_code mapped.')

##  Let's handle drugs as a binary question: we don't care how often
drugs_mapping = {'never':0, 'sometimes':1, 'often':1}
dataset['drugs_code'] = dataset.drugs.map(drugs_mapping)
print('drugs_code mapped.')

################################################################
##  Multiple Linear Regression

# We do seem to have a linear relationship between age and income,
# however it doesn't seem to be very precise at all.  Let's see
# if we can improve that by adding in some other features and
# running a multiple linear regression on the set.

x = dataset[['age', 'education_years', 'drinks_code',
             'smokes_code', 'drugs_code']]
print('x dataframe created')
y = dataset[['income']]
print('y dataframe created')

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state = 9)
print('train_test_split completed')
start = timeit.default_timer()
mult = LinearRegression()
mult.fit(x_train, y_train)
print('linear regression model fitted')

y_predict = mult.predict(x_test)
stop = timeit.default_timer()

print('Time to run multiple linear regression: ', end = '')
print(stop - start)

# Let's see how the correlation panned out

print(mult.coef_)
print('MLR Test Score: ', end = '')
print(mult.score(x_train, y_train))
print('MLR Train Score: ', end = '')
print(mult.score(x_test, y_test))

residuals = y_predict - y_test

plt.scatter(y_predict, residuals, alpha=0.4)
plt.title('Residual Analysis')
plt.show()

plt.scatter(y_test, y_predict, alpha = 0.2)
plt.xlabel('Income: $Y_i$')
plt.ylabel('Predicted Income: $\hat{Y}_i$')
plt.title('Actual Income vs Predicted Income')
plt.show()


################################################################
##  Classification


#  Let's see if we can predict an income bracket using other
#  variables in our dataset.

#  Brackets: 0-40K, 40K - 70K, 70K - 100K, 100K +

classifier = KNeighborsClassifier(n_neighbors = 5)

#  Create a new column for income bracket in our dataframe

def bracket_me (row):
    if row['income'] < 40001 :
        return 1
    if row['income'] < 70001 :
        return 2
    if row['income'] < 100000 :
        return 3
    return 4

dataset['income_bracket'] = dataset.apply (lambda row: bracket_me(row), axis=1)

cfr_dataset = dataset[['age', 'education_years', 'drinks_code',
             'smokes_code', 'drugs_code']]
cfr_labels = dataset[['income_bracket']]

X_train, X_test, y_train, y_test = train_test_split(cfr_dataset,
                                                  cfr_labels,
                                                  train_size = 0.8,
                                                  test_size = 0.2,
                                                  random_state = 9)

start = timeit.default_timer()
#  KNN classification:

knnClassifier = KNeighborsClassifier(n_neighbors = 5)
knnClassifier.fit(X_train, y_train.values.ravel())
stop = timeit.default_timer()

print('KNN Classifier test score:')
print(knnClassifier.score(X_test, y_test))
print('')
print('Time to run KNN classification: ', end = '')
print(stop - start)
print('')

start = timeit.default_timer()
#  SVM classification:

svmClassifier = SVC(gamma = 1)
svmClassifier.fit(X_train, y_train.values.ravel())
stop = timeit.default_timer()

print('SVM classifier test score:')
print(svmClassifier.score(X_test, y_test))
print('')
print('Time to run SVM classification: ', end = '')
print(stop - start)


