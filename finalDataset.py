import pandas as pd
import pathlib
import numpy as np

data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/datasets')

d1 = pd.read_csv(data_dir / 'processedTweets.csv', usecols=['Message','Target'])
d2 = pd.read_csv(data_dir / 'TweetsDataset.csv',usecols=['Message','Target'])
d3 = pd.read_csv(data_dir / 'tweetSentiment.csv',usecols=['Message','Target'])

dataset = pd.concat([d1,d2,d3])

dataset = dataset.sample(frac=1).reset_index(drop=True)

print(dataset)

#dataset.to_csv(data_dir / 'datasetFinal.csv')