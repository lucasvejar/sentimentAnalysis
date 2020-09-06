import pandas as pd
import pathlib
import numpy as np
from textblob import TextBlob


data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train')
save_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/datasets')

#tweet_sentiment
data_tweet_sentiment = pd.read_csv(data_dir / 'tweet_sentiment.csv')
dts = pd.DataFrame(data=data_tweet_sentiment.values, columns=data_tweet_sentiment.columns)

datasetF = pd.DataFrame(data=[tweet[1] for tweet in dts.values],columns=['Message'])

res = []

# 0 id 1 text 2 sentiment_main 3 sentiment
for d in dts.values:
    if (d[3]=='positive'):
        res.append(1)
    else:
        if (d[3] == 'negative'):
            res.append(0)
        else: 
            if (d[3]=='neutral' and (d[2] in ['empty','sadness','worry','hate'])): 
                res.append(0)
            else:
                res.append(1)

datasetF['Target'] = res

print(datasetF)

#datasetF.to_csv(save_dir / 'tweetSentiment.csv')