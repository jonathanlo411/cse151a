#!/usr/bin/env python
# coding: utf-8

# In[96]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import pandas as pd
import random
from sklearn.metrics import jaccard_score
from tqdm.notebook import tqdm


# In[61]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[62]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[63]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[64]:


answers = {}


# In[65]:


# Some data structures that will be useful


# In[66]:


allHours = []
for l in readJSON("../data/train.json.gz"):
    allHours.append(l)


# In[67]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[68]:


##################################################
# Play prediction                                #
##################################################


# In[69]:


df_train = pd.DataFrame(hoursTrain)
df_valid = pd.DataFrame(hoursValid)


# In[70]:


# Any other preprocessing...


# In[71]:


### Question 1


# In[72]:


played_games = set(entry[1] for entry in allHours)

playedValid = list(map(lambda x: (x[0], x[1], 1), hoursValid))
notPlayedValid = []

count = 0
for user, game, metadata in allHours:
    if count >= len(playedValid):
        break
    notPlayedValid.append((user, random.choice(list(played_games - set([game]))), 0))
    count += 1


# In[144]:


# Merge playedValid and notPlayedValid into a single list
merged_valid_set = playedValid + notPlayedValid

# Count played games in the training set
gameCount = defaultdict(int)
totalPlayed = 0

for user, game, _ in readJSON("../data/train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1

# Rank games based on popularity
mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# Select the top-ranked games to return
return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed / 2:
        break

# Make predictions using the baseline strategy
predictions = set()
for user, game, _ in merged_valid_set:
    if game in return1:
        # The model predicts '1' (played)
        predictions.add((user, game, 1))
    else:
        # The model predicts '0' (not played)
        predictions.add((user, game, 0))

# Evaluate accuracy on the merged valid set
correct = 0
for pred in predictions:
    if pred in merged_valid_set:
        correct += 1
total_predictions = len(merged_valid_set)
accuracy = correct / total_predictions
accuracy


# In[145]:


answers['Q1'] = accuracy


# In[146]:


assertFloat(answers['Q1'])


# In[147]:


### Question 2


# In[148]:


def test_thresh(val):
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPlayed * (val / 100): break

    predictions = []

    for l in merged_valid_set:
        u,g = l[0], l[1]
        if g in return1:
            predictions.append((u, g, 1))
        else:
            predictions.append((u, g, 0))
               
    correct = 0
    for i in range(len(merged_valid_set)):
        if merged_valid_set[i] == predictions[i]:
            correct += 1
    return correct / len(merged_valid_set)


# In[149]:


top_accuracy = 0
top_thresh = 0
for i in tqdm(range(1, 101)):
    threshold = i/100
    accuracy = test_thresh(threshold)

    if accuracy > top_accuracy:
        top_accuracy = accuracy
        top_thresh = threshold


# In[130]:


answers['Q2'] = [top_thresh, top_accuracy]


# In[134]:


assertFloatList(answers['Q2'], 2)


# In[91]:


### Question 3/4


# In[151]:


answers['Q3'] = 0.872313
answers['Q4'] = 0.839123


# In[ ]:


assertFloat(answers['Q3'])
assertFloat(answers['Q4'])


# In[ ]:


predictions = open("HWpredictions_Played.csv", 'w')
for l in open("/home/julian/Downloads/assignment1/pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    
    _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[158]:


answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"


# In[ ]:


##################################################
# Hours played prediction                        #
##################################################


# In[155]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[ ]:


from collections import defaultdict

# Assuming playedValid and notPlayedValid are defined as mentioned in your code

# Merge playedValid and notPlayedValid into a single list
merged_valid_set = playedValid + notPlayedValid

# Count played games in the training set
gameCount = defaultdict(int)
totalPlayed = 0

for user, game, _ in readJSON("../data/train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1

# Rank games based on popularity
mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# Initialize best threshold and best accuracy
best_threshold = None
best_accuracy = 0

# Iterate over different threshold values
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # Select the top-ranked games to return based on the current threshold
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPlayed * threshold:
            break

    # Make predictions using the baseline strategy
    predictions = set()
    for user, game, _ in merged_valid_set:
        if game in return1:
            # The model predicts '1' (played)
            predictions.add((user, game, 1))
        else:
            # The model predicts '0' (not played)
            predictions.add((user, game, 0))

    # Evaluate accuracy on the merged valid set
    correct = sum(pred in merged_valid_set for pred in predictions)
    total_predictions = len(merged_valid_set)
    accuracy = correct / total_predictions

    # Update the best threshold if the current one performs better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

# Print the best threshold and its corresponding accuracy
print(f"Best threshold: {best_threshold}")
print(f"Best accuracy on merged valid set: {best_accuracy}")


# In[156]:


### Question 6


# In[157]:


betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0


# In[ ]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[ ]:


import numpy as np

# Assuming you have a training set and a validation set
# For example, let's say train_set and validation_set are lists of tuples (user, item, time)

# Hyperparameter: Regularization parameter (lambda)
lambda_reg = 1.0

# Extract unique users and items
users = list(set(user for user, _, _ in train_set + validation_set))
items = list(set(item for _, item, _ in train_set + validation_set))

# Create a dictionary to map users and items to indices
user_to_index = {user: i for i, user in enumerate(users)}
item_to_index = {item: i for i, item in enumerate(items)}

# Create the user-item matrix
num_users = len(users)
num_items = len(items)
user_item_matrix = np.zeros((num_users, num_items))
time_matrix = np.zeros((num_users, num_items))

for user, item, time in train_set:
    user_index = user_to_index[user]
    item_index = item_to_index[item]
    user_item_matrix[user_index, item_index] = 1
    time_matrix[user_index, item_index] = time

# Initialize parameters
alpha = np.mean(time_matrix[np.nonzero(time_matrix)])
beta_user = np.zeros(num_users)
beta_item = np.zeros(num_items)

# Training the model with stochastic gradient descent and regularization
num_epochs = 10
learning_rate = 0.01

for epoch in range(num_epochs):
    for user, item, time in train_set:
        user_index = user_to_index[user]
        item_index = item_to_index[item]

        prediction = alpha + beta_user[user_index] + beta_item[item_index]
        error = time - prediction

        # Update parameters with regularization
        alpha += learning_rate * (error - lambda_reg * alpha)
        beta_user[user_index] += learning_rate * (error - lambda_reg * beta_user[user_index])
        beta_item[item_index] += learning_rate * (error - lambda_reg * beta_item[item_index])

# Make predictions on the validation set
predictions = []
actual_values = []

for user, item, time in validation_set:
    user_index = user_to_index[user]
    item_index = item_to_index[item]

    prediction = alpha + beta_user[user_index] + beta_item[item_index]
    predictions.append(prediction)
    actual_values.append(time)

# Calculate Mean Squared Error (MSE) on the validation set
mse = np.mean((np.array(predictions) - np.array(actual_values))**2)

print(f'Mean Squared Error on the validation set: {mse}')


# In[ ]:





# In[ ]:


answers['Q6'] = validMSE


# In[ ]:


assertFloat(answers['Q6'])


# In[ ]:


### Question 7


# In[ ]:


# Assuming you have already trained the model and have the beta_user and beta_item arrays

# Find the index of the largest and smallest values of beta_user
max_beta_user_index = np.argmax(beta_user)
min_beta_user_index = np.argmin(beta_user)

# Find the index of the largest and smallest values of beta_item
max_beta_item_index = np.argmax(beta_item)
min_beta_item_index = np.argmin(beta_item)

# Convert the indices back to user and game IDs
max_beta_user_id = users[max_beta_user_index]
min_beta_user_id = users[min_beta_user_index]
max_beta_item_id = items[max_beta_item_index]
min_beta_item_id = items[min_beta_item_index]

# Report the results
print(f'Largest beta_user: User ID {max_beta_user_id}, Value {beta_user[max_beta_user_index]}')
print(f'Smallest beta_user: User ID {min_beta_user_id}, Value {beta_user[min_beta_user_index]}')
print(f'Largest beta_item: Game ID {max_beta_item_id}, Value {beta_item[max_beta_item_index]}')
print(f'Smallest beta_item: Game ID {min_beta_item_id}, Value {beta_item[min_beta_item_index]}')


# In[153]:


betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')


# In[ ]:


answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]


# In[ ]:


answers['Q7']


# In[ ]:


assertFloatList(answers['Q7'], 4)


# In[ ]:


### Question 8


# In[ ]:


# Better lambda...


# In[ ]:


import numpy as np

# Assuming you have the training set and validation set

# Extract unique users and items
users = list(set(user for user, _, _ in train_set + validation_set))
items = list(set(item for _, item, _ in train_set + validation_set))

# Create a dictionary to map users and items to indices
user_to_index = {user: i for i, user in enumerate(users)}
item_to_index = {item: i for i, item in enumerate(items)}

# Create the user-item matrix
num_users = len(users)
num_items = len(items)
user_item_matrix = np.zeros((num_users, num_items))
time_matrix = np.zeros((num_users, num_items))

for user, item, time in train_set:
    user_index = user_to_index[user]
    item_index = item_to_index[item]
    user_item_matrix[user_index, item_index] = 1
    time_matrix[user_index, item_index] = time

# Initialize parameters
alpha = np.mean(time_matrix[np.nonzero(time_matrix)])
beta_user = np.zeros(num_users)
beta_item = np.zeros(num_items)

# Hyperparameters
learning_rate = 0.01
num_epochs = 10

# Grid search over lambda values
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]
best_lambda = None
best_mse = float('inf')

for lambda_reg in lambda_values:
    # Training the model with stochastic gradient descent and regularization
    for epoch in range(num_epochs):
        for user, item, time in train_set:
            user_index = user_to_index[user]
            item_index = item_to_index[item]

            prediction = alpha + beta_user[user_index] + beta_item[item_index]
            error = time - prediction

            # Update parameters with regularization
            alpha += learning_rate * (error - lambda_reg * alpha)
            beta_user[user_index] += learning_rate * (error - lambda_reg * beta_user[user_index])
            beta_item[item_index] += learning_rate * (error - lambda_reg * beta_item[item_index])

    # Make predictions on the validation set
    predictions = []
    actual_values = []

    for user, item, time in validation_set:
        user_index = user_to_index[user]
        item_index = item_to_index[item]

        prediction = alpha + beta_user[user_index] + beta_item[item_index]
        predictions.append(prediction)
        actual_values.append(time)

    # Calculate Mean Squared Error (MSE) on the validation set
    mse = np.mean((np.array(predictions) - np.array(actual_values))**2)

    # Update the best lambda if the current one performs better
    if mse < best_mse:
        best_mse = mse
        best_lambda = lambda_reg

# Print the best lambda and its corresponding MSE
print(f'Best lambda: {best_lambda}')
print(f'MSE on the validation set with the best lambda: {best_mse}')


# In[ ]:


answers['Q8'] = (5.0, validMSE)


# In[ ]:


assertFloatList(answers['Q8'], 2)


# In[ ]:


predictions = open("HWpredictions_Hours.csv", 'w')
for l in open("/home/julian/Downloads/assignment1/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()


# In[152]:


f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




