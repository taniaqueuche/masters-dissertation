#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
get_ipython().system('pip install textblob')
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
get_ipython().system('pip install nltk')
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords


# In[90]:


df = pd.read_csv('Personality_VTO.csv')
df = df[['Optional question. What’s your opinion on Virtual Try-On technology?']]
df = df.rename(columns={'Optional question. What’s your opinion on Virtual Try-On technology?':'review'})
df.dropna(how='all', inplace=True)
df.astype(str)

df['review']=df['review'].apply(str)
len(df)
df


# In[3]:


#exploring the dataset
sns.distplot(df['review'].str.len())


# # Tokenize

# In[129]:


tokenized = df['review']
tokenized = tokenized.apply(word_tokenize)
tokenized


# # Stopwords

# In[5]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stoplist = set(stopwords.words('english'))
stoplist = stoplist.update(['de', 'pas', 'le', 'et', 'pour'])


# In[103]:


from collections import Counter
pattern = "[A-Za-zÀ-ÿ0-9]+(?:'[st])?"

c = Counter()
df['review'].apply(lambda x: c.update(re.findall(pattern, x)))

# putting the result in a new DataFrame
vocab = pd.Series(list(c.values()), index=c.keys())
vocab = vocab.sort_values(ascending=False)
vocab.head(40)


# # method article
# https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk#step-2-%E2%80%94-tokenizing-the-data

# # Textblob
# https://www.kdnuggets.com/2018/08/emotion-sentiment-analysis-practitioners-guide-nlp-5.html

# In[152]:


# compute sentiment scores (polarity) and labels

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

df['sentiment_score'] = df['review'].apply(pol)
df['subjectivity'] = df['review'].apply(sub)
"""df['sentiment_label'] = ['positive' if score > 0.1 
                             else 'negative' if score < -0.1
                                 else 'neutral' 
                                     for score in df['sentiment_score']"""
df.loc[df["review"]=="I m convinced that is a good idea. People who may be reluctant to shop online is because they cannot try things on. We go to a physical shops to not be disappointed from online shopping and a potential gap between what we see in showroom and how it would fit with our face/body.  The Virtual Try-on answers to this problem. However some factors still lack between buying physically and online. We do not have any clue about the weight, the feeling of having this sun glasses on our nose. I am wondering if my hair will be stuck in the branches of the sun glasses or not. A point to be improved : the time of loading. "]


# In[75]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(y="sentiment_score", 
                   #hue='sentiment_label', 
                   data=df, ax=ax1)

bp = sns.boxplot(y="sentiment_score", 
                 #hue='sentiment_label', 
                 data=df, ax=ax2, palette="Set2")
t = f.suptitle('Visualizing Review Sentiment', fontsize=18)


# # Wordcloud
# https://www.datacamp.com/community/tutorials/wordcloud-python

# ### for text (not tokenized)

# In[11]:


text = " ".join(review for review in df.review)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[105]:


# Generate a word cloud image
wordcloud = WordCloud(stopwords=stoplist, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

"""# Generate a word cloud image TOKENIZED
wordcloud = WordCloud(stopwords=stoplist, background_color="white").generate(tokenized_str)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()"""


# ### for tokenized_str (tokenized)

# In[51]:


# using list comprehension 
tokenized_str = ' '.join(map(str, tokenized)) 
  
print(tokenized_str)  


# In[56]:


##next step: lemmatize
##next step: remove punctuation
punc = '''!()-[]{};:''"\,<>./?@#$%^&*_~'''
for ele in tokenized_str:  
    if ele in punc:  
        tokenized_str = tokenized_str.replace(ele, "")  
       
#remove None values to be able to iterate
from functools import partial
from operator import is_not
#filter(partial(is_not, None), tokenized_str)

##next step: remove stopwords
#tokenized_str = [word for word in tokenized_str if not word in stoplist]
tokenized_str


# ### looking for relevant pairs of words

# In[133]:


#recurring pairs
from nltk import ngrams
from collections import Counter

t = tokenized_str.split()
pairs = Counter(list(ngrams(t, 3)))
pairs.most_common()

