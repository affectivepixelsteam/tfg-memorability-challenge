import keras
import keras.backend as K

def CCC4keras_newVersion(y_true, y_pred):
    # extracted from : https://gitlab.com/snippets/1730605
    '''Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient

    The concordance correlation coefficient is the correlation between two variables that fall on the 45 degree line through the origin.

    It is a product of
    - precision (Pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation:
    - `rho_c =  1` : perfect agreement
    - `rho_c =  0` : no agreement
    - `rho_c = -1` : perfect disagreement

    Args:
    - y_true: ground truth
    - y_pred: predicted values

    Returns:
    - concordance correlation coefficient (float)
    '''

    # y_pred=K.cast(K.argmax(y_pred, axis=-1),dtype='float32')
    # y_true = K.cast(K.argmax(y_true, axis=-1),dtype='float32')
    # y_true = K.argmax(y_true, axis=-1)
    # print(y_pred)
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)
    s_xy = K.mean((y_true - x_m) * (y_pred - y_m))  # 1.0 / (N - 1.0 + K.epsilon()) *
    # condordance correlation coefficient
    ccc = K.mean((2.0 * s_xy) / (s_x_sq + s_y_sq + (x_m - y_m) ** 2))
    return 1 - (ccc)  # optimum when the CCC is equal to 1


def get_optim_and_lossFunct(optimizer = 'adam', learning_rate = 0.0001, momentum = 0.9, decay = 1e-6, lossFunction='cxentropy'):
    # OPTIMIZER AND LOSS FUNCTION
    if (optimizer == 'sgd'):
        # nesterov: boolean. Whether to apply Nesterov momentum.
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=0.9, nesterov=False)
    elif (optimizer == 'rms'):
        optim = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9)
    elif (optimizer == 'adam'):
        optim = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.1, decay=decay)
    elif (optimizer == 'adamMax'):
        optim = keras.optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.1, epsilon=1e-08, decay=decay)
    elif(optimizer=="adagrad"):
        optim = keras.optimizers.Adagrad(lr=learning_rate)
    else:
        optim = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=False)

    if (lossFunction == 'cxentropy'):  # RMSE is just the square root of variance, which is actually Standard Deviation.
        lossFunc = keras.losses.categorical_crossentropy
    elif (lossFunction == 'bxentropy'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.binary_crossentropy
    elif (lossFunction == 'mse'):  # MSE incorporates both the variance and the bias of the predictor.
        lossFunc = keras.losses.mean_squared_error
    elif (lossFunction == 'mae'):
        lossFunc = keras.losses.mean_absolute_error
    elif (lossFunction == 'CCC'):
        lossFunc = CCC4keras_newVersion
    else:
        lossFunc = keras.losses.mean_squared_error

    return optim, lossFunc


