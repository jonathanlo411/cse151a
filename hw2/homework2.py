#!/usr/bin/env python
# coding: utf-8

# In[54]:


import random
from sklearn import linear_model
from matplotlib import pyplot as plt
from collections import defaultdict
import gzip
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, jaccard_score
import matplotlib.pyplot as plt


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


answers = {}


# In[4]:


def parseData(fname):
    for l in open(fname):
        yield eval(l)


# In[5]:


data = list(parseData("./../data/beer_50000.json"))


# In[6]:


random.seed(0)
random.shuffle(data)


# In[7]:


dataTrain = data[:25000]
dataValid = data[25000:37500]
dataTest = data[37500:]


# In[8]:


yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
yValid = [d['beer/ABV'] > 7 for d in dataValid]
yTest = [d['beer/ABV'] > 7 for d in dataTest]


# In[9]:


categoryCounts = defaultdict(int)
for d in data:
    categoryCounts[d['beer/style']] += 1


# In[10]:


categories = [c for c in categoryCounts if categoryCounts[c] > 1000]


# In[11]:


catID = dict(zip(list(categories),range(len(categories))))


# In[12]:


df_train = pd.DataFrame(dataTrain)
df_valid = pd.DataFrame(dataValid)
df_test = pd.DataFrame(dataTest)


# review_columns = ['review/aroma', 'review/appearance', 'review/palate', 'review/taste', 'review/overall']<br>
# sample_df['reviews'] = df[review_columns].values.tolist()

# In[13]:


MAX_REVIEW_LENGTH = max(list(map(lambda x: len(x['review/text']), data)))

def apply_feat(df, includeCat = True, includeReview = True, includeLength = True):
    sample_df = pd.DataFrame()
    if includeCat:
        sample_df['beer/style'] = df['beer/style'].apply(lambda x: catID[x] if x in catID else "dummy")
        sample_df = pd.get_dummies(sample_df, columns=['beer/style']).drop(columns=['beer/style_dummy'])
    if includeReview:
        sample_df['review/aroma'] = df['review/aroma']
        sample_df['review/appearance'] = df['review/appearance']
        sample_df['review/palate'] = df['review/palate']
        sample_df['review/taste'] = df['review/taste']
        sample_df['review/overall'] = df['review/overall']
    if includeLength:
        sample_df['review_length'] = df['review/text'].apply(lambda x: len(x)/MAX_REVIEW_LENGTH)
    return sample_df


# In[14]:


### Question 1


# In[15]:


df_train_1 = apply_feat(df_train, True, False, False)
df_valid_1 = apply_feat(df_valid, True, False, False)
df_test_1 = apply_feat(df_test, True, False, False)


# In[16]:


model = linear_model.LogisticRegression(C=10, class_weight="balanced")
model.fit(df_train_1, yTrain)


# In[17]:


validBER = 1 - balanced_accuracy_score(yValid, model.predict(df_valid_1))
testBER = 1 - balanced_accuracy_score(yTest, model.predict(df_test_1))


# In[18]:


answers


# In[19]:


answers['Q1'] = [validBER, testBER]


# In[20]:


assertFloatList(answers['Q1'], 2)


# In[21]:


### Question 2


# In[22]:


df_train_2 = apply_feat(df_train)
df_valid_2 = apply_feat(df_valid)
df_test_2 = apply_feat(df_test)


# In[23]:


model = linear_model.LogisticRegression(C=10, class_weight="balanced", max_iter=1000)
model.fit(df_train_2, yTrain)


# In[24]:


validBER = 1 - balanced_accuracy_score(yValid, model.predict(df_valid_2))
testBER = 1 - balanced_accuracy_score(yTest, model.predict(df_test_2))


# In[25]:


answers['Q2'] = [validBER, testBER]


# In[26]:


assertFloatList(answers['Q2'], 2)


# In[27]:


### Question 3


# In[28]:


validBER_values = []
testBER_values = []
best_C = None
best_validBER = float('inf')
best_testBER = float('inf')
C_values = [0.001, 0.01, 0.1, 1, 10]
for c in C_values:
    model = linear_model.LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
    model.fit(df_train_2, yTrain)

    validBER = 1 - balanced_accuracy_score(yValid, model.predict(df_valid_2))
    testBER = 1 - balanced_accuracy_score(yTest, model.predict(df_test_2))
    
    validBER_values.append(validBER)
    testBER_values.append(testBER)

    if validBER < best_validBER:
        best_C = c
        best_validBER = validBER
        best_testBER = testBER

plt.figure(figsize=(8, 6))
plt.semilogx(C_values, validBER_values, label='Validation BER', marker='o')
plt.semilogx(C_values, testBER_values, label='Test BER', marker='o')
plt.xlabel('C (Inverse of Regularization Strength)')
plt.ylabel('Balanced Error Rate (BER)')
plt.legend()
plt.title('Validation and Test BER for Different C Values')
plt.grid(True)

plt.show()

print("Best C:", best_C)
print("Best Validation BER:", best_validBER)
print("Best Test BER:", best_testBER)


# In[29]:


answers['Q3'] = [best_C, best_testBER, best_validBER]


# In[30]:


assertFloatList(answers['Q3'], 3)


# In[ ]:


### Question 4


# In[43]:


# No cat
train_data = apply_feat(df_train, False, True, True)
test_data = apply_feat(df_test, False, True, True)
model = linear_model.LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
model.fit(train_data, yTrain)
testBER_noCat = 1 - balanced_accuracy_score(yTest, model.predict(test_data))


# In[44]:


# No review
train_data = apply_feat(df_train, True, False, True)
test_data = apply_feat(df_test, True, False, True)
model = linear_model.LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
model.fit(train_data, yTrain)
testBER_noReview = 1 - balanced_accuracy_score(yTest, model.predict(test_data))


# In[45]:


# No length
train_data = apply_feat(df_train, True, True, False)
test_data = apply_feat(df_test, True, True, False)
model = linear_model.LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
model.fit(train_data, yTrain)
testBER_noLength = 1 - balanced_accuracy_score(yTest, model.predict(test_data))


# In[46]:


answers['Q4'] = [testBER_noCat, testBER_noReview, testBER_noLength]


# In[47]:


assertFloatList(answers['Q4'], 3)


# In[ ]:


### Question 5


# In[40]:


path = "./../data/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz"
f = gzip.open(path, 'rt', encoding="utf8")

header = f.readline()
header = header.strip().split('\t')


# In[49]:


dataset = []

for line in f:
    fields = line.strip().split('\t')
    d = dict(zip(header, fields))
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    dataset.append(d)


# In[50]:


dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]


# In[65]:


target_item_id = 'B00KCHRKD6'

# Get the product title for the target item
target_item_title = [item['product_title'] for item in dataset if item['product_id'] == target_item_id][0]

# Calculate Jaccard similarity for all items with the target item
jaccard_similarities = []

for item in dataset:
    if item['product_id'] == target_item_id:
        jaccard_similarities.append((0, item['product_id']))  # Jaccard similarity with itself is 0
    else:
        set1 = set(target_item_title.split())
        set2 = set(item['product_title'].split())
        jaccard_similarity = len(set1.intersection(set2)) / len(set1.union(set2))
        jaccard_similarities.append((jaccard_similarity, item['product_id']))

# Sort the items by Jaccard similarity in descending order and select the top 10
top_10_similar_items = sorted(jaccard_similarities, key=lambda x: x[0], reverse=True)[:10]

# Display the top 10 similar items and their Jaccard similarities
for similarity, item_id in top_10_similar_items:
    print(f"Jaccard Similarity: {similarity}, Item ID: {item_id}")


# In[66]:


answers['Q5'] = ms = top_10_similar_items 


# In[68]:


assertFloatList([m[0] for m in ms], 10)


# In[ ]:


### Question 6


# In[ ]:


def MSE(y, ypred):
    # ...


# In[ ]:


def predictRating(user,item):
    # ...


# In[ ]:


alwaysPredictMean = 


# In[ ]:


simPredictions = 


# In[ ]:


labels = 


# In[ ]:





# In[ ]:


answers['Q6'] = MSE(simPredictions, labels)


# In[ ]:


assertFloat(answers['Q6'])


# In[ ]:


### Question 7


# In[ ]:


itsMSE = 1000


# In[ ]:


answers['Q7'] = ["Description of your solution", itsMSE]


# In[ ]:


assertFloat(answers['Q7'][1])


# In[69]:


f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


# In[ ]:




