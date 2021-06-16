import pandas as pd
import numpy as np

def label_encoder(target):
    """

    :return:
    """
    target = target
    try:
        label = list(np.where(target.iloc[:, 0] == 'Y', 1, 0))
        print("label encoder is successful")
        return pd.DataFrame(label)
    except:
        print("Error occured in label encoder method")