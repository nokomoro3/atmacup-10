
import numpy as np
from sklearn.metrics import mean_squared_error

def calc_loss(y_true, y_pred):
    return  np.sqrt(mean_squared_error(y_true, y_pred))
