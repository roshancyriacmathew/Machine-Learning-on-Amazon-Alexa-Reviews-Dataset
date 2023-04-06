#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv('amazon_alexa.tsv',sep='\t')
df.head()


# In[3]:


df.info()


# In[4]:


sns.countplot(x='rating', data=df)


# In[5]:


fig = plt.figure(figsize=(7,7))
colors = ("red","gold","yellowgreen","cyan","orange")
wp = {'linewidth':2, 'edgecolor':'black'}
tags = df['rating'].value_counts()
explode = (0.1,0.1,0.2,0.3,0.2)
tags.plot(kind='pie', autopct='%1.1f',colors=colors, shadow=True,
          startangle=0, wedgeprops=wp, explode=explode, label='')
plt.title('Distribution of the different ratings')
plt.show()


# In[6]:


fig = plt.figure(figsize=(30,7))
sns.countplot(x="variation",data=df)


# In[7]:


fig = plt.figure(figsize=(20,10))
sns.countplot(y="variation",data=df)


# In[8]:


df['variation'].value_counts()


# In[9]:


sns.countplot(x='feedback', data=df)
plt.show()


# In[10]:


fig = plt.figure(figsize=(7,7))
tags = df['feedback'].value_counts()
tags.plot(kind='pie', autopct='%1.1f%%', label='')
plt.title("Distribution of the different sentiments")
plt.show()


# In[11]:


for i in range(5):
    print(df['verified_reviews'].iloc[i],"\n")


# In[12]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+www\S+|https\S+", '', text, flags= re.MULTILINE)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[13]:


df.verified_reviews = df['verified_reviews'].apply(data_processing)


# In[14]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[15]:


df['verified_reviews'] = df['verified_reviews'].apply(lambda x: stemming(x))


# In[16]:


for i in range(5):
    print(df['verified_reviews'].iloc[i],"\n")


# In[17]:


pos_reviews = df[df.feedback == 1]
pos_reviews.head()


# In[18]:


text = ' '.join([word for word in pos_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews', fontsize=19)
plt.show()


# In[19]:


neg_reviews = df[df.feedback==0]
neg_reviews.head()


# In[20]:


text = ' '.join([word for word in neg_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize=19)
plt.show()


# In[21]:


X = df['verified_reviews']
Y = df['feedback']


# In[22]:


cv = CountVectorizer()
X = cv.fit_transform(df['verified_reviews'])


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[24]:


print("Size of x_train: ",(x_train.shape))
print("Size of y_train: ",(y_train.shape))
print("Size of x_test: ",(x_test.shape))
print("Size of y_test: ",(y_test.shape))


# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[27]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("Test accuracy: {:.2f}%".format(logreg_acc*100))


# In[28]:


print(confusion_matrix(y_test, logreg_pred))
print("\n")
print(classification_report(y_test, logreg_pred))


# In[29]:


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_pred = mnb.predict(x_test)
mnb_acc = accuracy_score(mnb_pred, y_test)
print("Test accuracy: {:.2f}%".format(mnb_acc*100))


# In[30]:


print(confusion_matrix(y_test, mnb_pred))
print("\n")
print(classification_report(y_test, mnb_pred))


# In[ ]:




