# tweepy imports
from tweepy import API, Cursor, OAuthHandler, Stream
from tweepy.streaming import StreamListener

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import credentials as twitter_credentials
import re
import string
import cleaningTweets


class TwitterClient():
    
    def __init__(self, twitter_user = None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user


    def get_twitter_client_api(self):
        return self.twitter_client


    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


class TwitterAuthenticator():
    
    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TwitterStreamer():
    # class for streaming and processing live tweets

    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()

    def stream_tweets(self, fetched_tweets_filename, hashtags_list):
        # this handles twitter authentication and the connection to twitter streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app()
        stream = Stream(auth, listener)

        # this line filter Twitter Streams to capture data by the keywords
        stream.filter(track = hashtags_list)


class TwitterListener(StreamListener):
    # this is a basic listener that just prints received tweets to stdout

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
    
    def on_error(self, status):
        if status == 420:
            # returning False on_data method in case rate limit occurs
            return False
        print(status)


class TweetAnalyzer():
    # functionality for analyzing and categorizing content from tweets

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array(([tweet.created_at for tweet in tweets]))
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df


    def plot_tweet_info(self, dataFrame):
        # Shows a grapich with tweets metrics
        time_likes = pd.Series(data=dataFrame['len'].values, index=dataFrame['date'])
        time_likes.plot(figsize=(16, 4), color='r', label='length', legend=True)
        
        time_favs = pd.Series(data=dataFrame['likes'].values, index=dataFrame['date'])
        time_favs.plot(figsize=(16, 4), color='b', label='likes', legend=True)
        
        time_retweets = pd.Series(data=dataFrame['retweets'].values, index=dataFrame['date'])
        time_retweets.plot(figsize=(16, 4), color='g', label='retweets', legend=True)
        plt.show()    
    
    def getTweetsCleaned(self,tweets):
        cleaner = cleaningTweets.Cleaner()
        return cleaner.prepare_data(tweets)


if __name__ == '__main__':

    twitter_client = TwitterClient('adrian_twarog')
    tweet_analyzer = TweetAnalyzer()

   # api = twitter_client.get_twitter_client_api()

    tweets = twitter_client.get_user_timeline_tweets(20)
    #print(dir(tweets[0]))
    #print(tweets[0].retweet_count)

    df = tweet_analyzer.tweets_to_data_frame(tweets)

    # Get average length over all tweets:
    print("Average length over all tweets: ",np.mean(df['len']))
    
    # Get the number of likes for the most liked tweet:
    print("Number of likes for the most liked tweet: ",np.max(df['likes']))
    
    # Get the number of retweets for the most retweeted tweet:
    print("Number of retweets for the most retweeted tweet: ",np.max(df['retweets']))
    
    # print the first 10 tweets
    top_10_tweets = df.head(10)
    print(top_10_tweets['tweets'])

    #clean = cleaningTweets.Cleaner()
    #cleaned = clean.prepare_data(top_10_tweets['tweets'])
    cleaned = tweet_analyzer.getTweetsCleaned(top_10_tweets['tweets'])
    print(cleaned)
    # tweet_analyzer.plot_tweet_info(df)