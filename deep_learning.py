import plac
import random
import pathlib
import cytoolz
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from spacy.compat import pickle
import spacy

import pandas as pd

import tensorflow as tf
from keras import backend as K


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
session = tf.Session(config=config)
K.set_session(session)


class SentimentAnalyser(object):
    
    @classmethod
    def load(cls,path,nlp,max_length):
        return "Some"

    def __init__(self, model, max_length=100):  
        self._model = model
        self.max_length = max_length
    
    def __call__(self, doc):
        x = "some"
        y = "some"


def get_labelled_sentences(docs, doc_labels): 
    labels = []

def get_features(docs, max_length):
    docs = list(docs)


def train(
    train_texts,
    train_labels,
    dev_texts,
    dev_labels,
    lstm_shape,
    lstm_settings,
    lstm_optimizer,
    batch_size=100,
    nb_epoch=5,
    by_sentence=True
):
    print("------>   Loading spaCy   <------------")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)



def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape["max_length"],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    model.add(
        TimeDistributed(Dense(shape["nr_hidden"],use_bias=False))
    )
    model.add(
        Bidirectional(
            LSTM(
                shape['nr_hidden'],
                recurrent_dropout = settings['dropout'],
                dropout= settings['dropout']
            )
        )
    )
    model.add(
        Dense(shape['nr_class'],activation='sigmoid'),
    )
    model.compile(
        optimizer= Adam(lr=settings['lr']),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )
    return model

def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model_dir, texts, labels, max_length=100):
    nlp = spacy.load("en_vectors_web_lg")


def read_data(data_dir, limit=0):
    dataset = pd.read_csv(data_dir / 'tweet_sentiment.csv')
    df = pd.DataFrame(data=dataset.values, columns=dataset.columns)
    nf = pd.DataFrame(data=[tweet for tweet in df['text']], columns=['Tweets'])        
    tweets = nf['Tweets']
    sentiments = []
    for sentiment in df['sentiment']:  
        sentiments.append(-1 if sentiment=='negative' else (1 if sentiment=='positive' else 0))
    
    examples = zip(tweets,sentiments)
    examples = list(examples)

    if limit >= 1:
        examples = examples[:limit]
    return zip(*examples)  # Unzips into two lists


def main(
    model_dir='/home/lv11/Documents/ProyectosPython/sentimentAnalysis/model_lstm',
    train_dir= pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train'),
    dev_dir= pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/test'),
    is_runtime=False,
    nr_hidden= 64,
    max_length = 100,
    dropout = 0.5,
    learn_rate = 0.001,
    nb_epoch = 5,
    batch_size = 256,
    nr_examples = -1
):
    # directory path
    model_dir = pathlib.Path(model_dir)
    # reading the data
    dev_texts, dev_labels = read_data(dev_dir)
    train_texts, train_labels = read_data(train_dir)
    #converting labels as array
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_labels = np.asarray(dev_labels, dtype="int32")

    lstm = train(
        train_texts,
        train_labels,
        dev_texts,
        dev_labels,
        {"nr_hidden": nr_hidden, "max_length": max_length, "nr_class": 1},
        {"dropout": dropout, "lr": learn_rate},
        {},
        nb_epoch= nb_epoch,
        batch_size = batch_size
    )


if __name__ == '__main__':
    plac.call(main)
    K.get_session().close()