from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from string import punctuation
import re


class Cleaner():

    def spacy_tokenizer(self, sentence):
        # load english tokenizer, tagger, parser, NER and word vectors
        parser = English()
        sentence  = self.cleaning_tweet(sentence)
        tokens = parser(sentence)
        tokens = [ word.lemma_.lower().strip() if word.lemma_ != '-PRON-' else word.lower_ for word in tokens ]
        tokens = [ word for word in tokens if word not in STOP_WORDS and word not in punctuation ]
        # tokens = list(tokens)  # removing the duplicates tokens ? 
        return tokens

    def cleaning_tweet(self, tweet):
        # removing html special entities
        tweet = re.sub(r'\&\w*;', '', tweet)
        # convert @username to AT_USER
        tweet = re.sub('@[^\s]+','',tweet)
        # remove stickers
        tweet = re.sub(r'\$\w*', '', tweet)
        # removing links
        tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
        # removing hashtags
        tweet = re.sub(r'#\w*', '', tweet)
        # Remove words with 2 or fewer letters
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        # Remove whitespace (including new line characters)
        tweet = re.sub(r'\s\s+', ' ', tweet)
        # Remove single space remaining at the front of the tweet.
        tweet = tweet.lstrip(' ') 
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
        tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
        # Removing numbers 
        tweet = re.sub(r'[0-9]+','',tweet)
        return tweet

    def prepare_data(self, tweets):
        return [ self.spacy_tokenizer(tweet) for tweet in tweets]


# This was a test code 
# if __name__ == '__main__':    
#    tweet = "@GolosoAlonso New #dotnotes! Today we're reviewing database fundamentals, including OLTP vs OLAP + data organization and 1 more. https://meet.google.com/rue. This is the new beggining"
#    print('original tweet: ',tweet)
 #   print('clean tweet: ',prepare_data(tweet))