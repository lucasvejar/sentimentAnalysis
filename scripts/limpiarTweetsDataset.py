import pandas as pd
import pathlib
import numpy as np
from textblob import TextBlob


data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train')
save_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/datasets')

# Loading the TweetsDataset
tweetsDataset = pd.read_csv(data_dir / 'tweetsDataset.csv')


dataset = pd.DataFrame(data=[tweet for tweet in tweetsDataset['Message']],columns=['Message'])
dataset['Target'] = tweetsDataset['Target']
res = []
for tweet in dataset.values:
    #print(tweet[1])
    res.append(0 if (tweet[1] == (-1)) else ( 0 if TextBlob(tweet[0]).sentiment[0] <= 0 else 1 ))
  #  if (tweet[1]==0):
  #      sent = TextBlob(tweet[0])
  #      tweet[1] = 0 if sent.sentiment[0] <= 0 else 1
  #  else:
  #      if (tweet[1]==-1):
  #          tweet[1] = 0

d = pd.DataFrame(data=[tweet for tweet in tweetsDataset['Message']],columns=['Message'])
d['Target'] = res 
print(d)

#d.to_csv(save_dir / 'TweetsDataset.csv')
