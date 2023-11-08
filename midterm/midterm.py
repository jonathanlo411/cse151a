#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import gzip
import math
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn import linear_model
import random
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


import warnings

warnings.simplefilter('ignore')


# In[ ]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[ ]:


answers = {}


# In[ ]:


z = gzip.open("./../data/train.json.gz")


# In[ ]:


dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)


# In[ ]:


z.close()


# In[ ]:


### Question 1


# In[ ]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])


# In[ ]:


df = pd.DataFrame(dataset)


# In[ ]:


df['text length'] = df['text'].str.len()
df['ones'] = 1


# In[ ]:


X = df[['ones', 'hours']]
y = df[['text length']]


# In[ ]:


mod = linear_model.LinearRegression()
mod.fit(X,y)
predictions = mod.predict(X) 


# In[ ]:


theta_1 = mod.coef_[0][1]


# In[ ]:


mse_q1 = mean_squared_error(y, predictions)


# In[ ]:


answers['Q1'] = [theta_1, mse_q1]


# In[ ]:


assertFloatList(answers['Q1'], 2)


# In[ ]:


### Question 2


# In[ ]:


median_hours = df['hours'].median()


# In[ ]:


X['log hours'] = df['hours'].transform(lambda x: math.log2(x + 1))
X['sq root hours'] = df['hours'].transform(lambda x: math.sqrt(x))
X['above median'] = df['hours'].transform(lambda x: x > median_hours)
X


# In[ ]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[ ]:


mse_q2 = mean_squared_error(y, predictions)


# In[ ]:


answers['Q2'] = mse_q2


# In[ ]:


assertFloat(answers['Q2'])


# In[ ]:


### Question 3


# In[ ]:


X = pd.DataFrame()
X['ones'] = df['ones']
X['lambda 1'] = df['hours'].transform(lambda x: x > 1)
X['lambda 2'] = df['hours'].transform(lambda x: x > 5)
X['lambda 3'] = df['hours'].transform(lambda x: x > 10)
X['lambda 4'] = df['hours'].transform(lambda x: x > 100)
X['lambda 5'] = df['hours'].transform(lambda x: x > 1000)
X


# In[ ]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[ ]:


mse_q3 = mean_squared_error(y, predictions)


# In[ ]:


answers['Q3'] = mse_q3


# In[ ]:


assertFloat(answers['Q3'])


# In[ ]:


### Question 4


# In[ ]:


X = df[['ones', 'text length']]
X


# In[ ]:


y = df[['hours']]


# In[ ]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)


# In[ ]:


mse = mean_squared_error(y, predictions)
mae = mean_absolute_error(y, predictions)


# In[ ]:


mse, mae


# In[ ]:


answers['Q4'] = [mse, mae, "MAE is better in this case because we are just trying to find the generalized result"]


# In[ ]:


assertFloatList(answers['Q4'][:2], 2)


# In[ ]:


### Question 5


# In[ ]:


y_trans = df['hours'].transform(lambda x: math.log2(x + 1))


# In[ ]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[ ]:


mse_trans = mean_squared_error(y_trans, predictions_trans)


# In[ ]:


predictions_untrans = list(map(lambda x: 2**x - 1, predictions_trans))


# In[ ]:


mse_untrans = mean_squared_error(y, predictions_untrans)


# In[ ]:


answers['Q5'] = [mse_trans, mse_untrans]


# In[ ]:


assertFloatList(answers['Q5'], 2)


# In[ ]:


### Question 6


# In[ ]:


def feat6(d):
    val = [0] * 100
    hours = int(d['hours'])
    if (hours > 99):
        val[99] = 1
    else:
        val[hours] = 1
    return val


# In[ ]:


X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]


# In[ ]:


Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]


# In[ ]:


alphas = [1, 10, 100, 1000, 10000]
best_alpha = None
best_mse_val = float('inf')
best_mse_test = None

for alpha in alphas:
    # Fit Ridge regression model
    model = linear_model.Ridge(alpha=alpha)
    model.fit(Xtrain, ytrain)

    # Predict on the validation set
    y_pred_val = model.predict(Xvalid)

    # Calculate MSE on validation set
    mse_val = mean_squared_error(yvalid, y_pred_val)

    if mse_val < best_mse_val:
        best_mse_val = mse_val
        best_alpha = alpha
        # Predict on the test set with the best model
        y_pred_test = model.predict(Xtest)
        best_mse_test = mean_squared_error(ytest, y_pred_test)


# In[ ]:


best_alpha, best_mse_val, best_mse_test


# In[ ]:


answers['Q6'] = [best_alpha, best_mse_val, best_mse_test]


# In[ ]:


assertFloatList(answers['Q6'], 3)


# In[ ]:


### Question 7


# In[ ]:


times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)


# In[ ]:


nNotPlayed = df['hours'].transform(lambda x: x < 1).sum()


# In[ ]:


answers['Q7'] = [median, nNotPlayed]


# In[ ]:


assertFloatList(answers['Q7'], 2)


# In[ ]:


### Question 8


# In[ ]:


X = df[['ones', 'text length']]
y = [d['hours_transformed'] > median for d in dataset]


# In[ ]:


mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions


# In[ ]:


def rates(predictions, y):
    TP = [a and b for (a,b) in zip(predictions,y)]
    TN = [not a and not b for (a,b) in zip(predictions,y)]
    FP = [a and not b for (a,b) in zip(predictions,y)]
    FN = [not a and b for (a,b) in zip(predictions,y)]

    TP = sum(TP)
    TN = sum(TN)
    FP = sum(FP)
    FN = sum(FN)
    
    return TP, TN, FP, FN


# In[ ]:


TP, TN, FP, FN = rates(predictions, y)


# In[ ]:


BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))


# In[ ]:


BER


# In[ ]:


answers['Q8'] = [TP, TN, FP, FN, BER]


# In[ ]:


assertFloatList(answers['Q8'], 5)


# In[ ]:


### Question 9


# In[ ]:


y_pred_prob = mod.predict_proba(X)[:, 1]  # Probabilities of being in the positive class
k_values = [5, 10, 100, 1000]
precisions_at_k = []

for k in k_values:
    # Sort predictions by predicted probability in descending order
    sorted_indices = np.argsort(-y_pred_prob)
    
    # Determine the threshold for the k-th ranked prediction
    threshold = y_pred_prob[sorted_indices[k - 1]]
    
    # Convert y and predictions to NumPy arrays
    y_np = np.array(y)
    predictions_np = np.array(predictions)
    
    # Find the indices of all predictions with probability >= threshold
    selected_indices = np.where(y_pred_prob >= threshold)[0]
    
    # Compute precision for these selected predictions
    precision = precision_score(y_np[selected_indices], predictions_np[selected_indices])
    
    precisions_at_k.append(precision)

# Print precision@k values
for i, k in enumerate(k_values):
    print(f'Precision@{k} = {precisions_at_k[i]:.4f}')


# In[ ]:


answers['Q9'] = precisions_at_k


# In[ ]:


assertFloatList(answers['Q9'], 4)


# In[ ]:


### Question 10


# In[ ]:


y_trans = [d['hours_transformed'] for d in dataset]


# In[ ]:


mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)


# In[ ]:


your_threshold = 3.7
predictions_thresh = list(map(lambda x: x > your_threshold, predictions_trans))
TP, TN, FP, FN = rates(predictions_thresh, y_trans)
BER = 0.5 * (FP / (TN + FP) + FN / (FN + TP))
BER


# In[ ]:


answers['Q10'] = [your_threshold, BER]


# In[ ]:


assertFloatList(answers['Q10'], 2)


# In[ ]:


### Question 11


# In[ ]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[ ]:


df_train = pd.DataFrame(dataTrain)
df_test = pd.DataFrame(dataTest)


# In[ ]:


userMedian = defaultdict(list)
itemMedian = defaultdict(list)

reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

ratingPerUser = defaultdict(list)
ratingPerItem = defaultdict(list)


for d in dataset:
    u,i,h_t = d['userID'],d['gameID'], d['hours_transformed']
    ratingPerUser[u].append(h_t)
    ratingPerItem[i].append(h_t)

for user in ratingPerUser:
    userMedian[user] = statistics.median(ratingPerUser[user])

for item in ratingPerItem:
    itemMedian[item] = statistics.median(ratingPerItem[item])

for user in ratingPerUser:
    userMedian[user] = statistics.median(ratingPerUser[user])

for item in ratingPerItem:
    itemMedian[item] = statistics.median(ratingPerItem[item])


# In[ ]:


answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]


# In[ ]:


assertFloatList(answers['Q11'], 2)


# In[ ]:


### Question 12


# In[ ]:


global_median = df_train['hours_transformed'].median()
def f12(u,i):
    if u in ratingPerUser:
        if i in ratingPerItem:
            if itemMedian[i] > global_median:
                return 1
        elif userMedian[u] > global_median:
                return 1
    return 0


# In[ ]:


preds = [f12(d['userID'], d['gameID']) for d in dataTest]


# In[ ]:


global_median


# In[ ]:


y = df_test['hours_transformed'] > global_median


# In[ ]:


accuracy = accuracy_score(y, preds)


# In[ ]:


accuracy


# In[ ]:


answers['Q12'] = accuracy


# In[ ]:


assertFloat(answers['Q12'])


# In[ ]:


### Question 13


# In[ ]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)


# In[ ]:


def Jaccard(s1, s2):
    numerator = len(s1. intersection (s2))
    denominator = len(s1.union(s2))
    return numerator / denominator


# In[ ]:


def mostSimilar (i, K): 
    similarities = []
    users = usersPerItem [i] # Users who have purchased i
    for j in usersPerItem : # Compute similarity against each
        if j == i: continue
        sim = Jaccard(users , usersPerItem [j])
        similarities .append ((sim ,j))
    similarities .sort(reverse=True) # Sort to find the most
    return similarities [:K]


# In[ ]:


ms = mostSimilar(dataset[0]['gameID'], 10)


# In[ ]:


answers['Q13'] = [ms[0][0], ms[-1][0]]


# In[ ]:


assertFloatList(answers['Q13'], 2)


# In[ ]:


### Question 14


# In[ ]:


def Cosine(i1, i2, ratingDict):
    # Between two items
    inter = usersPerItem[i1].intersection(usersPerItem[i2])
    numer = 0
    denom1 = 0
    denom2 = 0
    for u in inter:
        numer += ratingDict[(u, i1)] * ratingDict[(u, i2)]
    for u in usersPerItem[i1]:
        denom1 += ratingDict[(u, i1)]**2
    for u in usersPerItem[i2]:
        denom2 += ratingDict[(u, i2)]**2
    denom = math.sqrt(denom1) * math.sqrt(denom2)
    if denom == 0:
        return 0
    return numer / denom

def mostSimilar(i, K, ratingDict):
    similarities = []
    users = usersPerItem[i]  # Users who have purchased item i
    for j in usersPerItem:  # Compute similarity against each item
        if j == i:
            continue
        sim = Cosine(i, j, ratingDict)
        similarities.append((sim, j))
    similarities.sort(reverse=True, key=lambda x: x[0])  # Sort to find the most similar
    return similarities[:K]

# Initialize ratingDict based on your dataset
ratingDict = {}
for d in dataset:
    u, i = d['userID'], d['gameID']
    lab = 1 if d['hours_transformed'] > global_median else -1
    ratingDict[(u, i)] = lab

# Calculate the most similar items to the first gameID in the dataset
ms = mostSimilar(dataset[0]['gameID'], 10, ratingDict)


# In[ ]:


answers['Q14'] = [ms[0][0], ms[-1][0]]


# In[ ]:


assertFloatList(answers['Q14'], 2)


# In[ ]:


### Question 15


# In[ ]:


ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = d['hours_transformed']
    ratingDict[(u,i)] = lab


# In[ ]:


ms = mostSimilar(dataset[0]['gameID'], 10, ratingDict)


# In[ ]:


answers['Q15'] = [ms[0][0], ms[-1][0]]


# In[ ]:


assertFloatList(answers['Q15'], 2)


# In[ ]:


f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:


answers

