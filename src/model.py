import numpy as np
import pandas as pd
from best_model_finder import parameter_tuner
from data_preprocessing import preprocessing,utils

## Read dataset
train_dataset = pd.read_csv("../dataset/processed/standardized_train_dataset.csv")
test_dataset = pd.read_csv("../dataset/processed/standardized_test_dataset.csv")
train_target = pd.read_csv("../dataset/processed/train_labels.csv")
test_target = pd.read_csv("../dataset/processed/test_labels.csv")

## Best model
best_model = parameter_tuner.model_finder()

##label encoder

y_train = utils.label_encoder(train_target)
y_test = utils.label_encoder(test_target)

## best model

best_algo = best_model.get_best_Model(train_dataset, test_dataset, y_train, y_test)