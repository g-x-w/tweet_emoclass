# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:37:57 2022

@author: 16475
"""


#  check if text_to_word works
#get rid of emotions and puncturations
import csv
import keras
max_vocab=2000
sen=[]
lab=[]
'''
lol manually one hot encoding 
00001 sadness 
00010 happiness
00100 no specific emotion
01000 fear
10000 anger

'''

with open('xxx.csv', 'r') as csvfile:
     tweetreader = csv.reader(csvfile)
     firstline=True
     for sentence, label in tweetreader:
         if firstline:
             firstline=False
             continue
         else:
             sen.append(keras.text_to_word_senquence(sentence))
             if label=='sadness':
                 lab.append([0,0,0,0,1])
             elif label=='happiness':
                 lab.append([0,0,0,1,0])
             elif label=='no specific emotion':
                 lab.append([0,0,1,0,0])
             elif label=='fear':
                 lab.append([0,1,0,0,0])
             else:
                 lab.append([1,0,0,0,0])
            


lab=[0,0,0,1,0]
sen1=[]
lab1=[]
with open('data_anger.csv', 'r') as csvfile:
    treader=csv.reader(csvfile)
    firstline=True
    for i_d, t in treader:
        if firstline:
            firstline=False
            continue
        else:
            sen1.append(keras.text_to_word_sequence(t))
            lab1.append(lab)





