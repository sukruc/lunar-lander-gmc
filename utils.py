import pandas as pd

def smoothen(scores):
    return pd.Series(scores).rolling(20, min_periods=1).mean()
