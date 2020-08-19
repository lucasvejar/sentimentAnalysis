import pandas as pd
import pathlib
import numpy as np

data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train')
data_dir2 = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/preprocessed-twitter-tweets')

data_positive = pd.read_csv(data_dir2 / 'processedPositive.csv')
data_negative = pd.read_csv(data_dir2 / 'processedNegative.csv')
#data_neutral = pd.read_csv(data_dir2 / 'processedNeutral.csv')
dataset = pd.read_csv(data_dir / 'tweet_sentiment.csv')

dfp = pd.DataFrame(data=data_positive.values, columns=data_positive.columns)
dfn = pd.DataFrame(data=data_negative.values, columns=data_negative.columns)
#dfu = pd.DataFrame(data=data_neutral.values, columns=data_neutral.columns)
#df = pd.DataFrame(data=dataset.values, columns=dataset.columns)
#nf = pd.DataFrame(data=[ tweet for tweet in df['text'] ], columns=['Message'])        


#res = []
#for sentiment in df['sentiment']:
#    if (sentiment!='neutral'):
#        res.append(0 if sentiment=='negative' else 1)
#nf['Target'] = res

# Set the dataSet with a particular value
def transformDataset(dataFrame, dir, value):
    df = pd.DataFrame(data=[tweet for tweet in dataFrame], columns=['Message'])
    df['Target'] = [value for data in dataFrame]
    return df


x = transformDataset(dfp,data_dir2,1)
y = transformDataset(dfn,data_dir2,0)
#z = transformDataset(dfu,data_dir2,0)

nf = pd.concat([x,y])
#  Mixing the data from the DataFrames
nf = nf.sample(frac=1).reset_index(drop=True)

print(nf)
# this just save the dataset into another file
#nf.to_csv(data_dir / 'tweetsDataset1.csv')

# reading the dataset stored with the correct format
new = pd.read_csv(data_dir / 'tweetsDataset1.csv',skiprows=1,names=['Message','Target'])
print(new)