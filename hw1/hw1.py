# %%
import json
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, confusion_matrix, balanced_accuracy_score, precision_score
import numpy as np
import random
import gzip
import dateutil.parser
import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1

# %%
f = gzip.open("./../data/fantasy_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
# Get max review length
MAX_REVIEW_LENGTH = max(list(map(lambda x: len(x['review_text']), dataset)))
MAX_REVIEW_LENGTH

# %%
def feature(datum):
    return len(datum['review_text'])/MAX_REVIEW_LENGTH

# %%
# Filter down and add intercept
X = list(map(feature, dataset))
X = list(map(lambda x: [1, x], X))
Y = list(map(lambda x: x['rating'], dataset))

# %%
# Train model and obtain coef and MSE
reg = linear_model.LinearRegression().fit(X, Y)

theta = reg.coef_
MSE = mean_squared_error(Y, reg.predict(X))

# %%
answers['Q1'] = [theta[0], theta[1], MSE]

# %%
assertFloatList(answers['Q1'], 3)

# %%
### Question 2

# %%
for d in dataset:
    t = dateutil.parser.parse(d['date_added'])
    d['parsed_date'] = t

# %%
# Clean dataset
df = pd.DataFrame(dataset)
df['parsed_date'] = pd.to_datetime(df['parsed_date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
df['weekday'] = df['parsed_date'].dt.strftime('%A')
df['month'] = df['parsed_date'].dt.strftime('%B')
df['review_text_prop'] = list(map(lambda x: len(x['review_text'])/MAX_REVIEW_LENGTH, dataset))
one_hot_df = pd.get_dummies(df, columns=['weekday', 'month'], prefix=['weekday', 'month'])

one_hot_df = one_hot_df.drop(columns=['user_id', 'book_id', 'review_id', 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes', 'n_comments', 'review_text', 'parsed_date', 'weekday_Monday', 'month_January'])
df = df.drop(columns=['user_id', 'book_id', 'review_id', 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes', 'n_comments', 'review_text', 'parsed_date'])

# %%
# Obtain Features
X = one_hot_df[one_hot_df.columns.difference(['rating'])]
Y = one_hot_df['rating']

# %%
answers['Q2'] = [np.append(X.loc[0].values, [1]),np.append(X.loc[1].values, [1])]

# %%
assertFloatList(answers['Q2'][0], 19)
assertFloatList(answers['Q2'][1], 19)

# %%
### Question 3

# %%
# Clean datasets
df['weekday'] = LabelEncoder().fit_transform(df['weekday'])
df['month'] = LabelEncoder().fit_transform(df['month'])
df['intercept'] = 1
one_hot_df['intercept'] = 1

# %%
# Obtain features
direct_X = df[df.columns.difference(['rating'])]
direct_Y = df['rating']
one_hot_X = one_hot_df[one_hot_df.columns.difference(['rating'])]
one_hot_Y = one_hot_df['rating']

# %%
# Train models
direct_reg = linear_model.LinearRegression().fit(direct_X, direct_Y)
one_hot_reg = linear_model.LinearRegression().fit(one_hot_X, one_hot_Y)

# %%
# Obtain MSEs
mse2 = mean_squared_error(direct_Y, direct_reg.predict(direct_X))
mse3 = mean_squared_error(one_hot_Y, one_hot_reg.predict(one_hot_X))

# %%
answers['Q3'] = [mse2, mse3]

# %%
assertFloatList(answers['Q3'], 2)

# %%
### Question 4

# %%
random.seed(0)
random.shuffle(dataset)

# %%
# Re-clean everything
df = pd.DataFrame(dataset)
df['parsed_date'] = pd.to_datetime(df['parsed_date'], format='%Y-%m-%d %H:%M:%S%z', utc=True)
df['weekday'] = df['parsed_date'].dt.strftime('%A')
df['month'] = df['parsed_date'].dt.strftime('%B')
df['review_text_prop'] = list(map(lambda x: len(x['review_text'])/MAX_REVIEW_LENGTH, dataset))
one_hot_df = pd.get_dummies(df, columns=['weekday', 'month'], prefix=['weekday', 'month'])

df['weekday'] = LabelEncoder().fit_transform(df['weekday'])
df['month'] = LabelEncoder().fit_transform(df['month'])
df['intercept'] = 1
one_hot_df['intercept'] = 1

one_hot_df = one_hot_df.drop(columns=['user_id', 'book_id', 'review_id', 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes', 'n_comments', 'review_text', 'parsed_date', 'weekday_Monday', 'month_January'])
df = df.drop(columns=['user_id', 'book_id', 'review_id', 'date_added', 'date_updated', 'read_at', 'started_at', 'n_votes', 'n_comments', 'review_text', 'parsed_date'])

direct_X = df[df.columns.difference(['rating'])]
direct_Y = df['rating']
one_hot_X = one_hot_df[one_hot_df.columns.difference(['rating'])]
one_hot_Y = one_hot_df['rating']

# %%
train2, test2 = direct_X[:len(direct_X)//2], direct_X[len(direct_X)//2:]
train3, test3 = one_hot_X[:len(one_hot_X)//2], one_hot_X[len(one_hot_X)//2:]
trainY, testY = direct_Y[:len(direct_Y)//2], direct_Y[len(direct_Y)//2:]

# %%
# Train models
direct_reg = linear_model.LinearRegression().fit(train2, trainY)
one_hot_reg = linear_model.LinearRegression().fit(train3, trainY)

# %%
# Obtain MSEs
test_mse2 = mean_squared_error(testY, direct_reg.predict(test2))
test_mse3 = mean_squared_error(testY, one_hot_reg.predict(test3))

# %%
answers['Q4'] = [test_mse2, test_mse3]

# %%
assertFloatList(answers['Q4'], 2)

# %%
### Question 5

# %%
f = gzip.open("./../data/beer_50000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
df = pd.DataFrame(dataset)

# %%
df['binarized_rating'] = np.where(df['review/overall'] >= 4, 1, 0)
df['review_length'] = df['review/text'].apply(len)

# %%
X = df[['review_length']]
y = df['binarized_rating']
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(X, y)

# %%
y_pred = model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()
BER = 1 - balanced_accuracy_score(y, y_pred)

# %%
print("MSE for model:", mean_squared_error(y, y_pred))

# %%
answers['Q5'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q5'], 5)

# %%
### Question 6

# %%
K_values = [1, 100, 1000, 10000]
y_prob = model.predict_proba(X)[:, 1]
sorted_indices = np.argsort(-y_prob)
precs = []
for K in K_values:
    y_pred_at_K = y_prob[sorted_indices[:K]]
    y_true_at_K = y[sorted_indices[:K]]
    precision = precision_score(y_true_at_K, (y_pred_at_K >= 0.5).astype(int))
    precs.append(precision)

# %%
plt.plot(K_values, precs, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('K (log scale)')
plt.ylabel('Precision at K')
plt.title('Precision at K for Different K Values')
plt.grid(True)
plt.show()

# %%
answers['Q6'] = precs

# %%
assertFloatList(answers['Q6'], 4)

# %%
### Question 7

# %%
X = df[['review_length', 'beer/ABV', 'review/aroma']]
y = df['binarized_rating']

# %%
model = linear_model.LogisticRegression(class_weight='balanced')
model.fit(X, y)

# %%
y_pred = model.predict(X)
its_test_MSE = mean_squared_error(y, y_pred)

# %%
print("Model MSE is:", its_test_MSE)

# %%
answers['Q7'] = ["I reduced the model error by incorporating more features. Specifically, I added 'beer/ABV' and 'review/aroma' as I believed those would play a large factor in the overall review number.", its_test_MSE]

# %%
f = open("answers_hw1.txt", 'w')
f.write(str(answers) + '\n')
f.close()


