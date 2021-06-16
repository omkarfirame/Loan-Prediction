from data_preprocessing import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("../dataset/raw/train_ctrUa4K.csv")

preprocessor = preprocessing.Preprocessor(data)


## Separte target and features
dataset,target = preprocessor.seprate_label_feature(data, 'Loan_Status')

### Drop ID column
dataset = preprocessor.drop_ID_col(dataset)

## Impute numerical Missing data
numerical_feature_na = preprocessor.numerical_features_na(dataset)
refined_data = preprocessor.impute_numerical_na(dataset, numerical_feature_na)

## Impute catagorical Missing data

catagorical_na = preprocessor.catagorical_features_na(dataset)
refined_data = preprocessor.impute_catagorical_na(refined_data, catagorical_na)

## Encode dataset
encoded_data = preprocessor.encode_data(refined_data,5)

## train test Split
X_train,X_test,y_train,y_test = train_test_split(encoded_data,target,test_size=0.2,random_state=42)

## scale dataset

standardized_data_train = preprocessor.scale_dataset(X_train)
standardized_data_test = preprocessor.scale_dataset(X_test)

## Save processed dataset
standardized_data_train.to_csv("../dataset/processed/standardized_train_dataset.csv",index=False)
standardized_data_test.to_csv("../dataset/processed/standardized_test_dataset.csv",index=False)