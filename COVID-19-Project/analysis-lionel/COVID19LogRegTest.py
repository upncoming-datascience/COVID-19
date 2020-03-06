import numpy as np
import pandas as pd

# get data, clean
dataset = pd.read_csv("C:/Users/ldmag/Documents/GitHub/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/03-04-2020.csv")
selected = dataset.drop(['Province/State', 'Last Update', 'Latitude', 'Longitude'], axis=1)
selected = selected.rename(columns={"Country/Region": "Country"})

by_Confirmed = selected.groupby('Country')['Confirmed'].sum()
by_Confirmed = by_Confirmed.to_frame()
by_Deaths = selected.groupby('Country')['Deaths'].sum()
by_Deaths = by_Deaths.to_frame()
by_Recovered = selected.groupby('Country')['Recovered'].sum()
by_Recovered = by_Recovered.to_frame()
frames = [by_Confirmed, by_Deaths, by_Recovered]
by_All = pd.concat(frames, axis= 1, sort=False)

# exploratory analysis 
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x = by_All['Confirmed'])
by_All.plot(kind = 'bar', figsize = (20,15))

# correlation test
corr = by_All.corr()

'''
# scatter
fig, ax = plt.subplots(figsize = (20, 15))
ax.scatter(by_All.index, by_All.Confirmed)
ax.set_xlabel('Country')
ax.set_ylabel('Confirmed cases')
'''
'''
# Logistic Regression
X = by_All[['Confirmed', 'Deaths', 'Recovered']]
Y = by_All.index

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, max_iter = 500)
classifier.fit(X_train, Y_train)
'''

# Classifier does not converge, may need to scale country variable. 