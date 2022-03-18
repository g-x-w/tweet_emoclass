import numpy as np

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

def tweet_vectorize(tweet_label_in, glove_embeds):
    out_vectors = []

    tweet_list = tweet_label_in[0]
    label_list = tweet_label_in[1]

    for i in range(len(tweet_list)):
        curr_tweet = []
        while len(tweet_list[i]) > 0:
            curr_word = tweet_list[i].pop(0)
            if curr_word in glove_embeds:
                curr_tweet.append(glove_embeds[curr_word])
            else:
                curr_tweet.append(np.zeros(50))     # dimensionality of embedding vector
        curr_tweet = np.asarray(curr_tweet)
        out_vectors.append(curr_tweet)
    
    return (out_vectors, label_list)


if __name__ == "__main__":
    small_embeds = load_pretrained_embeddings("D:/OneDrive - University of Toronto/School/NSCI Y3/WINTER/ECE324/glove.6B/glove.6B.50d.txt")
    
    test = ([['hello', 'my', 'name', 'is', 'disaster', 'truck'], ['Fred','I','fucking','hate','you'], ['batman','went','to','new','fork','city']], [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])

    test_out = tweet_vectorize(test, small_embeds)
    print(test_out)
