import pandas as pd
import pathlib
import numpy as np
from textblob import TextBlob

data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train')
data_dir2 = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/preprocessed-twitter-tweets')
save_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/datasets')


# processedData twetts
data_positive = pd.read_csv(data_dir2 / 'processedPositive.csv')
data_negative = pd.read_csv(data_dir2 / 'processedNegative.csv')
data_neutral = pd.read_csv(data_dir2 / 'processedNeutral.csv')

dfp = pd.DataFrame(data=data_positive.values, columns=data_positive.columns)
dfn = pd.DataFrame(data=data_negative.values, columns=data_negative.columns)
dfu = pd.DataFrame(data=data_neutral.values, columns=data_neutral.columns)

#def transformDataset(dataFrame, dir):
#    res=[]
 #   df = pd.DataFrame(data=[tweet for tweet in dataFrame], columns=['Message'])
  #  for dt in df['Message']:
   #     sent = TextBlob(dt)
    #    res.append(1 if sent.sentiment[0] <= 0 else 0)
   # df['Target'] = res
   # return df

#dtsp = transformDataset(dfu,data_dir2)
#dtsp.to_csv(data_dir / 'neutralToPN.csv')

positivos = pd.DataFrame(data=[ tweet for tweet in dfp], columns=['Message'])
positivos['Target'] = [ 1 for tweet in positivos['Message']]
negativos = pd.DataFrame(data=[ tweet for tweet in dfn], columns=['Message'])
negativos['Target'] = [0 for tweet in negativos['Message']]
neutrales = pd.read_csv(data_dir / 'neutralToPN.csv',skiprows=1,names=['Message','Target'])


todos = pd.concat([positivos,negativos,neutrales])
todos = todos.sample(frac=1).reset_index(drop=True)
# hay que hacerlo aleatoria y mezclar los valores del dataset
print(todos)

#todos.to_csv(save_dir / 'processedTweets.csv')