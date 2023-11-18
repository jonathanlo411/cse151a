# Imports
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
from tqdm.notebook import tqdm
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
import yake
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import sys

def readJSON(path):
    f = gzip.open(path, 'rt', encoding='utf-8')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

allHours = []
for l in readJSON("./../data/train.json.gz"):
    allHours.append(l)

hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))

def Jaccard(s1, s2):
  numer = len(s1.intersection(s2))
  denom = len(s1.union(s2))
  if denom > 0:
      return numer/denom
  return 0

df = pd.DataFrame(list(map(lambda x: x[2], allHours)))
df.head(3)
df_train = df.iloc[:165000]
df_test = df.iloc[165000:]

kw_extractor = yake.KeywordExtractor()

df_games = df.groupby('gameID')['date'].count().to_frame().rename(columns={"date": 'count'})
df_games['avg_hours'] = df_train.groupby('gameID')['hours'].mean()
df_games.head(3)

df_users = df.groupby('userID')['date'].count().to_frame().rename(columns={"date": 'game_count'})
df_users['avg_hours'] = df_train.groupby('userID')['hours'].mean()
df_users.head(3)

# Groupby users and games and append all the texts
user_review_text = defaultdict(list)
game_review_text = defaultdict(list)
for u,g,d in allHours:
    user_review_text[u].append(d['text'])
    game_review_text[g].append(d['text'])

# Obtaining all keywords (Topcis) for all game reviews
game_topics = []
for game in tqdm(game_review_text):
    big_str = ' '.join(game_review_text[game])
    res = kw_extractor.extract_keywords(big_str)
    game_topics.append(list(map(lambda x: x[0], res))[:10])

# Obtaining all keywords (Topcis) for all reviews the user wrote
user_topics = []
for user in tqdm(user_review_text):
    big_str = ' '.join(user_review_text[user])
    res = kw_extractor.extract_keywords(big_str)
    user_topics.append(list(map(lambda x: x[0], res))[:10])

# Adding the extracted topics for the users and games
df_users['topic'] = user_topics
df_games['topic'] = game_topics

predictions = open("predictions_Played.csv", 'w')
for l in tqdm(open("./../data/pairs_Played.csv"), total=20000):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    maxSim = 0
    users = set(hoursPerItem[g])
    for g2,_ in hoursPerUser[u]:
        sim = Jaccard(users,set(hoursPerItem[g2]))
        if sim > maxSim:
            maxSim = sim

    try:
        user_topics = set(df_users.loc[u]['topic'])
        game_topics = set(df_games.loc[g]['topic'])
        topic_sim = Jaccard(user_topics,game_topics)
    except KeyError as e:
        topic_sim = 0
    pred = 0
    if (maxSim > 0.19 or len(hoursPerItem[g]) > 62 or topic_sim > 0.4) or (maxSim > 0.13 and topic_sim > 0.3):
        pred = 1
    predictions.write(u + ',' + g + ',' + str(pred) + '\n')
predictions.close()


# Any other preprocessing...
gamesPerUser = defaultdict(set)
usersPerGame = defaultdict(set)

train_games = set()
playersPerGame = defaultdict(int)
negative_set = []
totalPlayed = 0

for user,game, data in hoursTrain:
    gamesPerUser[user].add(game)
    usersPerGame[game].add(user)
    train_games.add(game)
    playersPerGame[game] += 1
    totalPlayed+= 1

train_games = list(train_games)

for d in hoursValid:
    user = d[0]
    not_match = [d for d in train_games if d not in gamesPerUser[user]]
    rand = random.randint(1, len(not_match)) - 1
    game = not_match[rand]
    negative_set.append([user,game, None])

mostPopular = [(playersPerGame[x], x) for x in playersPerGame]
mostPopular.sort()
mostPopular.reverse()

hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
hours_pergameuser = defaultdict(int)

for d in hoursTrain:
    hoursPerItem[d[1]].append(d[2]['hours_transformed'])
    hoursPerUser[d[0]].append(d[2]['hours_transformed'])
    hours_pergameuser[d[0], d[1]] = (d[2]['hours_transformed'])

def iterate(alpha, betaU, betaI, lamb):
    # Run alpha
    alpha = sum([
        hours_pergameuser[user,game] - (betaU[user] + betaI[game])
          for  user,game,_ in hoursTrain
        ])
    alpha = alpha / len(hoursTrain)

    # Run betaU
    for user in gamesPerUser:
        betaU[user] = sum([
            hours_pergameuser[user, game] - (alpha + betaI[game]) 
            for game in gamesPerUser[user]
          ])
        betaU[user] = betaU[user] / (len(gamesPerUser[user]) + lamb)

    # Run betaI
    for game in usersPerGame:
        betaI[game] = sum(
            hours_pergameuser[user, game] - (alpha + betaU[user])
              for user in usersPerGame[game]
            )
        betaI[game] = betaI[game] / (len(usersPerGame[game]) + lamb)

    value = 0
    for d in hoursTrain:
        value += alpha + betaU[d[0]] + betaI[d[1]] - hours_pergameuser[d[0], d[1]]
    for u in betaU:
        value += betaU[u]**2
    for i in betaI:
        value += betaI[i]**2
    return value


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)
alpha = globalAverage 

alpha = globalAverage
betaU = {}
betaI= {}
lowest = sys.float_info.max
for u in hoursPerUser:
    betaU[u] = 0

for g in hoursPerItem:
    betaI[g] = 0

best_betaU = betaU.copy()
best_betaI = betaI.copy()
best_alpha = alpha
lamda_q8 = 4.0
for i in tqdm(range(500)):
    train_val = iterate(alpha, betaU, betaI, lamda_q8)
    #print(train_val)
    if (train_val < lowest):
        best_alpha = alpha
        best_betaU  = betaU.copy()
        best_betaI = betaI.copy()
        lowest = train_val 

def get_ui(user, game):
    return [betaU[user], betaI[game]]

predictions = open("predictions_Hours.csv", 'w')
for l in open("./../data/pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    bu, bi = get_ui(u,g)
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()