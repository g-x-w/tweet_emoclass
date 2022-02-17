from http import client
from matplotlib.style import use
import pandas as pd
import numpy as np
from sklearn.linear_model import TweedieRegressor
import tweepy as tp

client = tp.Client(
    #secrets
)

test_path = "D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/120321-V11/tweetid_userid_keyword_sentiments_emotions_Argentina.csv/tweetid_userid_keyword_sentiments_emotions_Argentina.csv"

def csv_to_df(csv_filename, start_row, num_rows, read_column_list):
    df = pd.read_csv(csv_filename, skiprows=range(1, start_row), nrows=num_rows, usecols=read_column_list)
    df = pd.read_csv(csv_filename, usecols=read_column_list)
    return (df)

def get_tweets(df_in):
    id_list = []
    for i in df_in.index:
        id_list.append(df_in.loc[i].tweet_id)
    tweets = client.get_tweets(ids=id_list, user_auth=True)
    return (tweets)




if __name__ == "__main__":
    col_list = ['tweet_id', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity', 'emotion']
    df1 = csv_to_df(test_path, 100, num_rows=10, read_column_list=col_list)
    tweets = get_tweets(df1)

    for i in range(len(tweets.data)):
        print(tweets.data[i].id)
