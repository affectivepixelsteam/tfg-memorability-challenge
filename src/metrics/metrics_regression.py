import numpy as np
import keras.backend as K
import scipy

def PearsonCorrelation4keras(y_true, y_pred): #CC FOR KERAS
    #normalise
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/bottom

    return K.mean(result)


def CCC4Keras(y_pred, y_true):
    K.print_tensor(y_true, message='y_true = ')
    pc = PearsonCorrelation4keras(y_true, y_pred)
    devP = K.std(y_pred, axis=0)
    devT = K.std(y_true, axis=0)
    meanP = K.mean(y_pred, axis=0)
    meanT = K.mean(y_true, axis=0)
    powMeans = K.pow(meanP-meanT,2)

    varP = K.var(y_pred, axis=0)
    varT = K.var(y_true, axis=0)

    numerator = 2*pc*devP*devT
    denominator = varP+varT+powMeans
    CCC = numerator/denominator
    return K.sum(CCC)


#--------------------------FUNCTIONS NOT FOR KERAS---------------------------------------
def PC_numpy(y_pred, y_true):
    pred = y_pred.reshape(-1)
    true = y_true.reshape(-1)
    PC = np.corrcoef(pred, true)[0, 1]
    return PC

def PC_sklearn(y_pred, y_true):
    pred = y_pred.reshape(-1)
    true = y_true.reshape(-1)
    PC = scipy.stats.pearsonr(pred, true)[0]
    return PC

#Concordance Correlation Coefficient
def CCC(y_true, y_pred):
    #pred = np.asarray(y_pred).reshape(-1)
    pearson_coeff = np.corrcoef(y_pred, y_true)[0,1]
    variance_labels = np.var(y_true)
    variance_pred = np.var(y_pred)
    mean_labels = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    stand_labels = np.std(y_true)
    stand_pred = np.std(y_pred)
    mean_power = np.power((mean_pred-mean_labels), 2)
    CCC = (2*pearson_coeff*stand_labels*stand_pred)/(variance_pred+variance_labels+mean_power)
    return CCC
#-------------------------------------------------------------------------------------------
