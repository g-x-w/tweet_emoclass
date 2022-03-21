# -*- coding: utf-8 -*-




import csv
import keras
max_vocab=2000
sen=[]
lab=[]
'''
lol manually one hot encoding 
00001 sadness 
00010 happiness
00100 no emotion
01000 fear
10000 anger

'''

with open('data1.csv', 'r',encoding='utf-8') as csvfile:
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
            

#print(sen)
#print(lab)

lab=[1,0,0,0,0]
sen1=[]
lab1=[]
with open('data_anger.csv', 'r', encoding='utf-8') as csvfile:
    treader=csv.reader(csvfile)
    firstline=True
    for i_d, t in treader:
        if firstline:
            firstline=False
            continue
        else:
            sen1.append(keras.preprocessing.text.text_to_word_sequence(t.encode('ascii', 'ignore').decode('ascii')))
            lab1.append(lab)

#print(sen1)
#print(lab1)


