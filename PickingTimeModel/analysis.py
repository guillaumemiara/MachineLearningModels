__author__ = 'guillaumemiara'

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


# Load csv into dataframe
print "Loading data"
print 50*'-'
p_data = pd.read_csv('picking_data.csv')
o_data = pd.read_csv('order_items_data2.csv')

# Get familiar with data structure. Print first row

print "Picking data first row"
print p_data.iloc[:1]
print 50*'-'
print "Order data first row"
print o_data.iloc[:1]
print 50*'-'

'''
Notes to self

Data available
--------------
Picking data contains information about the trip visit of a shopper in a shop
We know
- how many customers orders the shopper has to prepare ( deliveries_count )
- how many items total the shoppers has to get ( I assume here that it is the sum for all orders in trip )
- who is the shopper ( age, experience, experience in this shop)
- How much times it takes at the cashier counter ( could depend on # items, orders, time of the day)


Order item data
We know for each ordered item
- at which trip it was picked
- some info about the order ( dept, id , whether or not the item is often in stock, if items needs to be weighed )

Problem setting
---------------

We want to model the time it takes for a person to shop in the store given its new trip order.
The business case as I understand it, is that we want to be able to estimate
- based on the historical data
- based on new customers' orders aggregated in a "trip order"
How much it would take for a shopper to complete a the new trip order in a shop.

Intuition
---------
The time it would take for a shopper to complete a new trip depends on
- how many items it has to pick
- how familiar the shopper is with the shop and with picking
- how many different customer's order he has ( imagining here that the shopper could be first doing all orders, and then go to cashier)
- how many departments the shopper will need to visit
how often the shopper needs to call the customer for missing items
- how often the picker needs to weigh an item its trip
- the busyness at the counter ( might depend on time of the day)

Approach
--------

A linear regression model seems like the natural thing to try first here. We want a continuous output.

1.SIMPLE MODEL ( simple.py )

It is not that easy of a problem.
I want to be pragmatic and have something to deliver within the 3 hours so I will take a first simple approach.
Let's make a first model where we will assume that:
- the store
- the experience of picker in the store
- the delivery counts
- the items counts
will help us to estimate the time for a new order


2. REFINE MODELS

Then after this first step, we can go into something more complex and take advantage of other data we have in a second time
Take into consideration
- The items and items info in the new order
- The time of the orders
...

I will discuss this in the latest part of 'simple.py'

'''

# Some DataAnalysis to get an idea of what we are looking at.

# p_data['time_in_store'] = p_data['delivery_started_at'] - p_data['first_item_picked_at']

#The following reveals that the timestamps are saved as str, we need to change that to time format
print "The data type for timestamps column is " + str(type(p_data.delivery_started_at[1]))
# Conversion
p_data['delivery_started_at'] = pd.to_datetime(p_data['delivery_started_at'])
p_data['first_item_picked_at'] = pd.to_datetime(p_data['first_item_picked_at'])
print  "The data type for timestamps column is now " + str(type(p_data.delivery_started_at[1]))


# Create a new column for time in store per trip
p_data['time_in_store'] = (p_data.delivery_started_at-p_data.first_item_picked_at).astype('timedelta64[m]')

p_data = p_data[p_data.time_in_store < 180]
# Let's do some plots to validate our thinking

fig = plt.figure()

p_data.plot(kind='hexbin', x='items_count', y='time_in_store', gridsize=50)

p_data.plot(kind='hexbin', x='deliveries_count', y='time_in_store', gridsize=50)

plt.show()

'''
The plot do show that
- the larger the number of items, the longer in store
- the higher the number of deliveries, the longer in store
'''

