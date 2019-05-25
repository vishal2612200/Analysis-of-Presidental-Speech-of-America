#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:32:22 2019

@author: vishal
"""

import pandas as pd 

import plotly
import matplotlib.pyplot as plt
from textblob import TextBlob 
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
%matplotlib inline
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data = pd.read_csv("commencement_speeches.csv") 
# Preview the first 5 lines of the loaded data 
data.head()



# for cleaning purpose of null value from building and room ucomment below one
#data = data[~data['building'].isnull()]
#data = data[~data['room'].isnull()]



data.president_name.value_counts()

#bar chart for top5 president_name that have gave maximum speech
topcategory = data.president_name.value_counts().head(5).to_dict()
group_data = list(topcategory.values())
group_names = list(topcategory.keys())
fig, ax = plt.subplots()
ax.barh(group_names,group_data)

#pie chart for  president_name that have gave maximum speech
topcategory = data.president_name.value_counts().to_dict()
group_data = list(topcategory.values())
group_names = list(topcategory.keys())
plt.pie(group_data, labels=group_names, startangle=90, autopct='%.2f%%')
plt.title('pie chart for type column with level upto 2 decimal')
plt.show()



data.city.value_counts()
# pie chart for top six city for presidential speech
topcategory = data.city.value_counts().head(6).to_dict()
group_data = list(topcategory.values())
group_names = list(topcategory.keys())
plt.pie(group_data, labels=group_names, startangle=90, autopct='%.2f%%')
plt.title('pie chart for type column with level upto 2 decimal')
plt.show()


#for testing purpose
data.building.head()

data.building.value_counts()
data.president.value_counts()
#graph for ploting number of speech by particular president
data.groupby('president_name').count()['president'].sort_values(ascending=False).plot(kind='bar',
                                                           title='Bar chart of president_name')
#boxplot graph with president_name and president
an =data.groupby('president_name').count()['president'].sort_values(ascending=False)
plt.boxplot(an)
plt.show()
# distplot graph for different states of USA
topcategory = data.state.value_counts().to_dict()
group_data = list(topcategory.values())
group_names = list(topcategory.keys())
seaborn.distplot(group_data,bins = 10)



