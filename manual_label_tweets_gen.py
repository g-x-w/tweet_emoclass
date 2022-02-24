

import tweepy
import csv

client =tweepy.Client(bearer_token="", return_type=dict)
query_anger="(mad OR annoyed OR pissed OR idiotic OR stupid OR frustrating) lang:en -is:retweet"
query_happy="(excited OR awesome OR fantastic OR amazing OR great OR lovely) lang:en -is:retweet"
query_sad="(depressed OR disappointed OR sad OR misearable OR hopeless OR unhappy) lang:en -is:retweet"
query_fear="(scared OR freaky OR terrifying OR shocking OR horrific OR panic) lang:en -is:retweet"





obj=client.search_recent_tweets(query_anger,max_results=100)
print(obj)
with open('data_anger.csv','w',newline='',encoding='utf-8') as file:
    csvwriter=csv.writer(file)
    csvwriter.writerow(['id','text'])
    key_list=['id','text']
    for i in range(100):
        csvwriter.writerow(((obj.get('data'))[i]).get(x) for x in key_list)

















