import pandas as pd
import pathlib
import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split



# Loading the dataset 
data_dir = pathlib.Path('/home/lv11/Documents/ProyectosPython/sentimentAnalysis/train')
nf = pd.read_csv(data_dir / 'tweetsDataset1.csv',skiprows=1,names=['Message','Target'])



nlp = English()
stop_words = list(STOP_WORDS)
#print(STOP_WORDS)

def spacy_tokenizer(sentence):
    tokens = nlp(sentence)
    tokens = [ word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens ]
    tokens = [ word for word in tokens if word not in stop_words and word not in punctuation ]
    return tokens

class predictors(TransformerMixin):
    
    def transform(self,x, **transform_params):
        return [ clean_text(text) for text in x ]

    def fit(self, x, y=None, **fit_params):
        return self
    
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()


vectorizer = CountVectorizer(tokenizer=spacy_tokenizer,ngram_range=(1,1))
classifier = LinearSVC(dual=False)

tfvectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)

x = nf['Message']
ylabels = nf['Target']

X_train, X_test, y_train, y_test = train_test_split(x, ylabels, test_size=0.2, random_state=42)

pipe = Pipeline(
    [
        ('cleaner', predictors()),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ]
)

pipe.fit(X_train, y_train)

test_prediction = pipe.predict(X_test) 
for (sample, prediction) in zip(X_test, test_prediction):
    print(sample," PREDICTION =====> ",prediction)

print("Accuracy test: ",pipe.score(X_test,y_test))
print("Accuracy: ",pipe.score(X_test,test_prediction))

print("Accuracy train: ",pipe.score(X_train,y_train))

tweet = ["That play was shit","that's the dumbiest idea ever","you're not the brighest but I can manage it"]
print(pipe.predict(tweet))