import pandas as pd
import numpy as np
from pyparsing import delimited_list
import tweepy as tp
import csv as csv
import time as tt

client = tp.Client(
    # API Keys here
)

test_path = "D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/120321-V11/tweetid_userid_keyword_sentiments_emotions_Argentina.csv/tweetid_userid_keyword_sentiments_emotions_Argentina.csv"

full_path = "D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/120321-V11/COVID19_twitter_full_dataset.csv/COVID19_twitter_full_dataset.csv"

file_out_path = "df_out2.csv"

def csv_to_df(csv_filename, start_row, num_rows, read_column_list):
    df = pd.read_csv(csv_filename, skiprows=range(1, start_row), nrows=num_rows, usecols=read_column_list)
    df = df.drop_duplicates(subset='tweet_id')
    return (df)

def get_tweets(df_in):
    id_list = []
    for i in df_in.index:
        id_list.append(df_in.loc[i].tweet_id)
    tweets = client.get_tweets(ids=id_list, user_auth=True)
    return (tweets)

def returned_ids(tweet_data):
    ret = []
    for i in range(len(tweet_data.data)):
        ret.append(tweet_data.data[i].id)
    return  (ret)

if __name__ == "__main__":
    col_list = ['tweet_id', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity', 'emotion']
    output_header = ['tweet_id', 'tweet_text', 'fear_intensity', 'anger_intensity', 'happiness_intensity', 'sadness_intensity', 'emotion']
    tweet_ct = 39600

    with open(file_out_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(output_header)

        while (tweet_ct < 100000):
            print('Getting tweets {} to {}'.format(tweet_ct, tweet_ct+100))
            df_main = csv_to_df(full_path, tweet_ct+1, num_rows=100, read_column_list=col_list)
            tweets = get_tweets(df_main)
            id_list = returned_ids(tweets)

            for i in range(len(id_list)):
                writer.writerow([str(id_list[i]), tweets.data[i].text, 
                df_main.loc[df_main['tweet_id'] == id_list[i]].fear_intensity.item(), 
                df_main.loc[df_main['tweet_id'] == id_list[i]].anger_intensity.item(), 
                df_main.loc[df_main['tweet_id'] == id_list[i]].happiness_intensity.item(), 
                df_main.loc[df_main['tweet_id'] == id_list[i]].sadness_intensity.item(),
                df_main.loc[df_main['tweet_id'] == id_list[i]].emotion.item()])
        
            tweet_ct += 100
            print('\tSleeping for 105sec \t current time: {}'.format(tt.ctime()))
            tt.sleep(105)