#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# In[5]:


ds = pd.read_csv(r"C:\Users\user\Desktop\finalProject\product.csv")




# In[6]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['product_description'])


# In[7]:


cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}


# In[10]:


for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['product_uid'][i]) for i in similar_indices]

    results[row['product_uid']] = similar_items[1:]
    
print('done!')


# In[11]:


def item(id):
    return ds.loc[ds['product_uid'] == id]['product_description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
        print("-------")

recommend(item_id=11, num=15)



pickle.dump(tfidf_matrix, open('content.pkl','wb'))
# Loading model to compare the results
collaborative = pickle.load(open('content.pkl','rb'))




