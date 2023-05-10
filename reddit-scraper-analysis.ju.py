# %% 

import os
import praw # reddit library for scraping/crawler

reddit = praw.Reddit(client_id=os.environ.get('CLIENT_ID'),
                     client_secret=os.environ.get('CLIENT_SECRET'),
                     user_agent=os.environ.get('USER_AGENT')
                     )

subreddit = reddit.subreddit('worldnews')

comments = []

for comment in subreddit.comments(limit=10000):
    comments.append(comment.body)

# %%

import spacy 
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

def label_sentiment(polarity):
    if polarity > 0:
        return 'positive' 
    elif polarity < 0:
        return 'negative' 
    else:
        return 'neutral'

sentiments = []
labels = []

for comment in comments:
    doc = nlp(comment)
    #print(doc._.blob.polarity)                            # Polarity: -0.125
    #print(doc._.blob.subjectivity)                        # Subjectivity: 0.9
    #print(doc._.blob.sentiment_assessments.assessments)   # Assessments: [(['really', 'horrible'], -1.0, 1.0, None), (['worst', '!'], -1.0, 1.0, None), (['really', 'good'], 0.7, 0.6000000000000001, None), (['happy'], 0.8, 1.0, None)]
    #print(doc._.blob.ngrams())
    #print('-----')
    sentiments.append(doc._.blob.polarity)
    labels.append(label_sentiment(doc._.blob.polarity))


df = pd.DataFrame({'label': labels, 'comment': comments, 'sentiment': sentiments})
df

# %% 

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.histplot(df['sentiment'],kde=True)

# %% 
sns.countplot(x=df["label"])
