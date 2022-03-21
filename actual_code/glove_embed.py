from lib2to3.pgen2 import token
import numpy as np
import csv
import keras
import time as tt

# def runtime(starttime):

def load_pretrained_embeddings(filepath):
    print("Loading in pretrained GloVe embeddings from: {}".format(filepath))
    
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as embedfile:
        for line in embedfile:
            split_line = line.split()
            token = split_line[0]
            embed = np.array(split_line[1:], dtype=np.float64)
            embeddings[token] = embed
    
    print("{} words loaded into embedding".format(len(embeddings)))
    
    return (embeddings)

def tokenize_csv(filename):
    sen=[]
    lab=[]

    with open(filename, mode='r',encoding='utf-8') as csvfile:
        tweetreader = csv.reader(csvfile)
        firstline=True
        for sentence, label in tweetreader:
            if firstline:
                firstline=False
                continue
            else:
                sen.append(keras.preprocessing.text.text_to_word_sequence(sentence.encode('ascii', 'ignore').decode('ascii')))
                if label=='sadness':
                    lab.append([0,0,0,0,1])
                elif label=='happiness':
                    lab.append([0,0,0,1,0])
                elif label=='no emotion':
                    lab.append([0,0,1,0,0])
                elif label=='fear':
                    lab.append([0,1,0,0,0])
                else:
                    lab.append([1,0,0,0,0])

    return (sen, lab)

def tokenize_manual(filename, label):
    '''
        00001 sadness 
        00010 happiness
        00100 no emotion
        01000 fear
        10000 anger
    '''

    sen=[]
    lab=[]
    with open(filename, mode='r', encoding='utf-8') as csvfile:
        treader=csv.reader(csvfile)
        firstline=True
        for i_d, t in treader:
            if firstline:
                firstline=False
                continue
            else:
                sen.append(keras.preprocessing.text.text_to_word_sequence(t.encode('ascii', 'ignore').decode('ascii')))
                lab.append(label)
    
    return (sen, lab)

def tweet_vectorize(tweet_label_in, glove_embeds):
    out_vectors = []

    tweet_list = tweet_label_in[0]
    label_list = tweet_label_in[1]

    for i in range(len(tweet_list)):
        curr_tweet = []
        while len(tweet_list[i]) > 0 and len(curr_tweet) < 50:
            curr_word = tweet_list[i].pop(0)
            if curr_word in glove_embeds:
                curr_tweet.append(glove_embeds[curr_word])
            else:
                curr_tweet.append(np.zeros(50))     # dimensionality of embedding vector
        if len(curr_tweet) < 50:
            while len(curr_tweet) < 50:
                curr_tweet.append(np.zeros(50))
        # else:
        #     print('Full length tweet, no zero-padding applied')

        curr_tweet = np.asarray(curr_tweet)
        out_vectors.append(curr_tweet)
    
    return (out_vectors, label_list)


if __name__ == "__main__":
    small_embeds = load_pretrained_embeddings("D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/glove.6B/glove.6B.50d.txt")
    training_set = [[], []]

    # for i in range(1, 6):
    #     train_tokens = tokenize_csv("D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/tweet_emotion/tweets_labels/data{}.csv".format(str(i)))
    #     temp_vec = tweet_vectorize(train_tokens, small_embeds)
    #     training_set[0] = training_set[0] + temp_vec[0]
    #     training_set[1] = training_set[1] + temp_vec[1]
    #     print('Finished vectorizing csv number {}'.format(str(i)), "Dataset size: ", len(training_set[0]), len(training_set[1]))

    manual_names = ['anger', 'fear', 'happy', 'sad']
    manual_labels = [[1,0,0,0,0], [0,1,0,0,0], [0,0,0,1,0], [0,0,0,0,1]]

    manual_set = [[], []]
    for i in range(len(manual_labels)):
        manual_tokens = tokenize_manual("D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/tweet_emotion/manually_labeled/data_{}.csv".format(manual_names[i]), manual_labels[i])
        temp_vec = tweet_vectorize(manual_tokens, small_embeds)
        manual_set[0] = manual_set[0] + temp_vec[0]
        manual_set[1] = manual_set[1] + temp_vec[1]
    
    print(manual_set)