__author__ = 'guillaumemiara'

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import linear_model

from sklearn.linear_model import Ridge

from sklearn.cross_validation import train_test_split

try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
except ImportError:
    from sklearn_backports import PolynomialFeatures
    from sklearn_backports import make_pipeline


print "A first approach to the problem with a linear regression"
print 50*'-'

"""
Data import
"""

# Load csv into dataframe
frame = pd.read_csv('picking_data.csv')

#Data preparation

#Conversion to time
frame['delivery_started_at'] = pd.to_datetime(frame['delivery_started_at'])
frame['first_item_picked_at'] = pd.to_datetime(frame['first_item_picked_at'])
# Adding column time in store
frame['time_in_store'] = (frame.delivery_started_at- frame.first_item_picked_at).astype('timedelta64[m]')

print "Data conversion done"

# Data normalization ( for better linear regression results is not necessary here because scikit has a built in function)

"""
Data analysis
"""

# Prepare model for scikit
X= frame.as_matrix(['deliveries_count','items_count','shopper_previous_store_trips_count','shopper_previous_trips_count'])
y= frame.as_matrix(['time_in_store'])

# Simple cross validation ( Training / Test )
# Note : Improvement will be to use Training/ Cross Validation / Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)

# Fit the data

# Step 1 is to choose the best polynomial degree for the regression
print 50*'-'
print "Step 1: choosing polynomial degree"

train_errors_deg = {}
CV_errors_deg = {}

for degree in range(10):
    print "Test for degree " + str(degree)
    for alpha in [0]:
        est = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha, normalize = True))
        est.fit(X_train, y_train)
        train_errors_deg[degree] = est.score(X_train,y_train)
        CV_errors_deg[degree] = est.score(X_test,y_test)

#Store errors in dic
#df_train_errors = DataFrame(train_errors).transpose()
#df_CV_errors = DataFrame(CV_errors).transpose()

#print train_errors_deg
#print CV_errors_deg


best_degree = max(CV_errors_deg, key=CV_errors_deg.get)
print " The polynomial degree that maximizes the Ridge regression coefficient is " + str(best_degree)
print " The test regression coefficient is " + str(CV_errors_deg[best_degree])
print "The associated training error is " + str(train_errors_deg[best_degree])


# We see from the output that the degree that gives the best regression coefficient is 2

# Step 2 is to choose the best alpha regularization term for the regression to prevent underfitting and overfitting

print 50*'-'
print "Step 2: choosing regularization term"
train_errors_alpha = {}
CV_errors_alpha = {}

for degree in [best_degree]:
    for alpha in [ 0.01, 0.1 , 0.2, 0.4, 1.6, 3.2, 6.4, 12.8]:
        print "Test for alpha " + str(alpha)
        est = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha, normalize = True))
        est.fit(X_train, y_train)
        train_errors_alpha[alpha] = est.score(X_train,y_train)
        CV_errors_alpha[alpha] = est.score(X_test,y_test)

# print train_errors_alpha
# print CV_errors_alpha

best_alpha = max(CV_errors_alpha, key=CV_errors_alpha.get)
print " The alpha term that maximizes the Ridge regression coefficient is " + str(best_alpha)
print " The test regression coefficient is " + str(CV_errors_alpha[best_alpha])
print "The associated training error is " + str(train_errors_alpha[best_alpha])

'''
Discussion on improvements

This model is very simple and just a first approach
Many things could be done to improve it.

1) One very first thing - the fitting of this model
-  I could have used a better cross validation method with a division Train/CV/Test
   ( This implementation was easier)
-  I could have tried other linear regression models ( LASSO for example)
-  I could have tried other alpha parameters

But that will not be enough, obviously, looking at the regression coefficient, I am not doing a great job here
( Should be closer to 1)

2) On the simple model, I would try using other features in the model
- I could account for the store ( some could be bigger than other, have btter time at cashier,etc..)
To do so, I would add as many column as store and put a 0 or 1 if the trip is in this store
- I could account for the shopper id
Same as above
- I could account for the shopper age
Adding feature "Date - shopper birth year"
- I could acount for the global shopper experience
USing shopper_previous_trip_count

And then see how it improves, analyzes the features to see which are the onw with more influence

3) The next step for improvement would be to think harder and account for the content of the new order

a.  A baby step is to break down the feature "items account" from the simple model
New features can be computed from order data

With order data, we can come up with:
- "items often missing"
- "items often found"
Computed from item_previous_ordered_count & item_previous_found_count

Then break down with
- "items needing weigh"
- "items not needing weigh"

b. Another step is to breakdown the items per department

etc...

Overall the idea is that I would refine the simple linear model with new features and measure how I improve on my error

'''

