import numpy as np
import pandas as pd
from best_model_finder import parameter_tuner
from data_preprocessing import preprocessing

## Read dataset
train_dataset = pd.read_csv("../dataset/processed/standardized_train_dataset.csv")
test_dataset = pd.read_csv("../dataset/processed/standardized_test_dataset.csv")
train_target = pd.read_csv("../dataset/processed/train_labels.csv")
test_target = pd.read_csv("../dataset/processed/test_labels.csv")

## Best model
best_model = parameter_tuner.model_finder()

##label encoder
y_train = pd.Series(np.where(train_target.iloc[:, 0] == 'Y', 1, 0))
y_test = pd.Series(np.where(test_target.iloc[:, 0] == 'Y', 1, 0))
#naive_bays = best_model.get_best_param_for_naive_bays(train_dataset, test_dataset, y_train, y_test)

## best model

best_algo = best_model.get_best_Model(train_dataset, test_dataset, y_train, y_test)