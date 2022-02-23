import pandas as pd
from csv import writer

def encode(input_filename, output_filename):
    df = pd.read_csv(input_filename, usecols=["tweet_text", "fear_intensity", "anger_intensity", "happiness_intensity", "sadness_intensity"])
    emotions = ['fear', 'anger', 'happiness', 'sadness']
    for num, entry in df.iterrows():
        scores = []
        scores.append(entry["fear_intensity"])
        scores.append(entry["anger_intensity"])
        scores.append(entry["happiness_intensity"])
        scores.append(entry["sadness_intensity"])
        max_score = max(scores)
        tie = False
        # two values with exact same score
        if scores.count(max_score) > 1:
            tie = True
        # check if any other scores are within the tolerance, if yes the tweet emotion is classified as 'no emotion'
        else:
            for score in scores:
                if score != max_score and (max_score - score) <= 0.01:
                    tie = True
        # constuct the output vector
        output_label = ''
        if tie == True:
            output_label = 'no emotion'
        else:
            ind_of_max = scores.index(max_score)
            output_label = emotions[ind_of_max]
        # read to the output csv file
        with open(output_filename, 'a', newline='', encoding='utf-8') as file:
            new_elem = [entry["tweet_text"], output_label]
            writer_file = writer(file)
            writer_file.writerow(new_elem)


if __name__ == "__main__":
    encode("df_out1.csv", "data1.csv")
    encode("df_out2.csv", "data2.csv")
    encode("df_out3.csv", "data3.csv")
    encode("df_out4.csv", "data4.csv")
    encode("df_out5.csv", "data5.csv")