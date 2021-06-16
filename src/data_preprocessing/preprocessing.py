import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """
    This class shall be used to clean and transform the dataset
    """
    
    def __init__(self,data):
        self.data = data

    def remove_column(self,data,column):
        """

        :param data:
        :param column:
        :return:
        """
        self.column = column
        try:
            self.refined_data = self.data.drop(labels=self.column,axis=1)
            print("Column removal successful")
            return self.refined_data
        except:
            print("Exception occures innremove column method of Preprocessor class")

    def seprate_label_feature(self,data,label_column_name):
        """

        :param data:
        :param label_column_name:
        :return:
        """
        self.col_name = label_column_name
        try:
            self.target = self.data[self.col_name]
            self.X = self.data.drop(labels=self.col_name,axis=1)
            print("Seprate label feature successful")
            return self.X,self.target
        except:
            print("Exception occures innremove column method of Preprocessor class")

    def catagorical_features_na(self,dataset):
        """

        :param dataset: dataframe to check catagorical features
        :return: list containiing names of catagorical features
        """
        self.dataset = dataset
        try:
            self.features_na = [feature for feature in self.dataset.columns if
                           self.dataset[feature].isnull().sum() > 1 and self.dataset[feature].dtypes == 'O']
            print("catagorical features na method successful")

            return self.features_na
        except:
            print("Exception occures in impute catagorical feature na method of Preprocessor class")

    def numerical_features_na(self,dataset):
        """

        :param dataset: dataframe to check numerical features
        :return: list containiing names of numerical features
        """
        # Numerical features
        self.dataset = dataset
        try:

            self.features_na = [feature for feature in self.dataset.columns if
                           self.dataset[feature].isnull().sum() > 1 and self.dataset[feature].dtypes != 'O']
            print("numerical_features_na successful")

            return self.features_na
        except:
            print("Exception occures in numerical feature na method of Preprocessor class")

    # check for date time variables
    def temporal_feature(self,dataset):
        """

        :param dataset: dataframe to check temporal features
        :return: list containiing names of temporal features
        """
        self.dataset = dataset
        try:
            self.temporal_feature = [feature for feature in self.dataset.columns if
                                'Yr' in feature or 'yr' in feature or 'Year' in feature or 'year' in feature or 'Date' in feature or 'date' in feature]
            return self.temporal_feature
        except:
            print("Exception occures in temporal feature method of Preprocessor class")

    # check for descrete variables
    def descrete_feature(self,dataset, numerical_features, temporal_feature):
        """

        :param dataset: dataframe to check temporal features
        :param numerical_features: list of numerical features
        :param temporal_feature: list of temporal features
        :return: list containing names of descrete features
        """
        self.dataset = dataset
        self.numerical_features = numerical_features
        self.temporal_feature = temporal_feature
        try:

            self.descrete_features = [feature for feature in self.numerical_features if
                                 len(self.dataset[feature].unique()) < 10 and feature not in self.temporal_feature]
            return self.descrete_features
        except:
            print("Exception occures in descrete feature method of Preprocessor class")

    def continous_feature(self,dataset, numerical_features, descrete_features):
        """

        :param dataset: dataframe to check temporal features
        :param numerical_features: list of numerical features
        :param descrete_features: list of descrete features
        :return: list of continous features
        """
        self.dataset = dataset
        self.numerical_features = numerical_features
        self.descrete_features = descrete_features

        # check for continuous variables
        try:

            self.continuous_features = [feature for feature in self.numerical_features if
                                   feature not in self.descrete_features + self.temporal_feature]
            print("Continous feature method successful")
            return self.continuous_features
        except:
            print("Exception occures in continous feature method of Preprocessor class")

    def drop_ID_col(self,dataset):
        """

        :param dataset: Combined dataset
        :return: dataframe without ID columns
        """
        self.dataset=dataset
        try:
            self.id_col = [feature for feature in self.dataset.columns if 'id' in feature or 'ID' in feature]
            self.dataset = self.dataset.drop(self.id_col, axis=1)
            print("drop ID col method successful")
            return self.dataset
        except:
            print("Exception occures in dropID column method of Preprocessor class")

    def impute_numerical_na(self,dataset, numerical_features_na):
        """


        :param dataset:
        :param numerical_features_na:
        :return:
        """
        self.dataset = dataset
        self.numerical_features_na = numerical_features_na
        try:
            for feature in self.numerical_features_na:
                self.median_value = self.dataset[feature].median()
                self.dataset[feature].fillna(self.median_value, inplace=True)
            print("Impute numerical na method successfull")
            return self.dataset
        except:
            print("Exception occures in impute numerical na method of Preprocessor class")

    def impute_catagorical_na(self,dataset, catagorical_features_na):
        """


        :param dataset:
        :param numerical_features_na:
        :return:
        """
        self.dataset = dataset
        self.catagorical_features_na = catagorical_features_na
        try:
            for feature in self.catagorical_features_na:
                self.dataset[feature].fillna("Missing", inplace=True)
            print("Impute catagorical na method successfull")
            return self.dataset
        except:
            print("Exception occures in impute numerical na method of Preprocessor class")

    def encode_data(self,dataset, threshold):
        """

        :param dataset:
        :param threshold:
        :return:
        """
        self.dataset = dataset
        self.threshold = threshold
        try:
            self.encode_data = [feature for feature in self.dataset.columns if len(list(self.dataset[feature].unique())) <= self.threshold]
            self.data = pd.get_dummies(self.dataset,columns=self.encode_data)
            print("Encode data method successful")
            return self.data
        except:
            print("Exception occures encode data method of Preprocessor class")


    def scale_dataset(self,data):
        """

        :param data:
        :return:
        """
        self.data = data
        try:
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(self.data)
            self.scaled_data_df = pd.DataFrame(data=self.scaled_data,columns=self.data.columns)
            print("Scale dataset method successfull")
            return self.scaled_data_df
        except:
            print("Exception occured in data scaling")
