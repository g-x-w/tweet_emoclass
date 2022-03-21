# -*- coding: utf-8 -*-
"""splitting_and_start_to_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vP_PBv2hRCxmxSmIECxxuG3HXe50T3SM
"""

import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle

def split_train_val_test(input, labels, man_input, man_lab):

  input_shuff, labels_shuff = shuffle(input, labels)

  training_proportion = 0.8
  validation_proportion = 0.1
  num_train = int(len(input_shuff) * training_proportion)
  num_val = int(len(input_shuff) * validation_proportion)

  input_train, input_valid, input_test = input_shuff[:num_train], input_shuff[num_train:num_train+num_val], input_shuff[num_train+num_val:]
  label_train, label_valid, label_test = labels_shuff[:num_train], labels_shuff[num_train:num_train+num_val], labels_shuff[num_train+num_val:]

  input_test += man_input
  label_test += man_lab

  return input_train, input_valid, input_test, label_train, label_valid, label_test

data = [[np.array([0, 1, 2]), np.array([3, 4, 5])], [np.array([4, 4, 4]), np.array([6, 6, 6])]]
test = [[np.array([7, 7, 7]), np.array([8, 8, 8])], [np.array([9, 9, 9]), np.array([10, 10, 10])]]
X, y = shuffle(data, test)
print(X)
print(y)


class model(nn.Module):
  def __init__(self):
    super(model, self).__init__()

    self.convolution_layer = nn.Conv2d(in_channels=1, out_channels=32, stride=1, kernel_size = 3)

  def forward(self, x):
    x = self.convolution_layer(x)
    x = nn.functional.relu(x)
    x = nn.functional.max_pool2d(x)