import pandas as pd
import numpy as np

def label_encoder(target):
    """

    :return:
    """
    try:
        #label = list(np.where(target.iloc[:, 0] == 'Y', 1, 0))
        target = target.iloc[:,0].map({"Y": 1, "N": 0})
        print("label encoder is successful")
        return pd.DataFrame(target)
    except:
        print("Error occured in label encoder method")