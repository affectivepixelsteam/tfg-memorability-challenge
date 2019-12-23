
# Implement a basic LSTM model to predict memorability from saliency
import pandas as pd
import numpy as np
import numpy.random as rand
import math, os, random
import datetime

import tensorflow as tf
from tensorflow import set_random_seed

import keras
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adagrad
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
import src.utils.nw_config_functions as nw_config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

import matplotlib.pyplot as plt
import src.metrics.metrics_regression as metrics_regression
from scipy.stats import spearmanr

config_ses=tf.ConfigProto()
config_ses.gpu_options.allow_growth=True
#config_ses.gpu_options.per_process_gpu_memory_fraction=0.8
config_ses.allow_soft_placement=True
config_ses.log_device_placement=False
for NUM_NEURONS in [50,100,150,200]:
    # Hyperparameters
    SEQUENCE_LENGTH = 168
    EMBEDDING_LENGTH = 84
    EPOCHS = 75#15
    BATCH_SIZE = 224#
    VALIDATION_SPLIT = 0.2
    COMPILER = 'rms' #adagrad, rmsprop
    LEARNING_RATE = 0.0001
    #NUM_NEURONS = 70
    LOSS = 'CCC' #bxentropy
    METRICS = ['mse', metrics_regression.PearsonCorrelation4keras,metrics_regression.CCC4Keras] #accuracy
    classification = False
    NORMALIZATION = False
    SEED = 10
    # SAME SEED
    random.seed(SEED)
    rand.seed(SEED)
    set_random_seed(SEED)


    # Path to file
    train_embeddings_file_path = '../../data/corpus/devset/dev-set/train_saliency_embeddings_splitted.csv'
    test_embeddings_file_path = '../../data/corpus/devset/dev-set/test_saliency_embeddings_splitted.csv'
    train_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/train_ground-truth_dev-set_splitted.csv'
    test_ground_truth_file_path = '../../data/corpus/devset/dev-set/ground-truth/test_ground-truth_dev-set_splitted.csv'
    save_processed_files = "../../data/corpus/devset/dev-set/numpy_files/"

    # Path model and weights
    model_save_path = '../../models/saliency/saliency-lstm-model_NEWCRIS.json'
    weight_save_path = '../../models/saliency/saliency-lstm-weight_NEWCRIS.h5'
    current_time = (datetime.datetime.now()).strftime("%Y%M%d_%H%M%S")
    log_dir = '../../data/LOGS/saliency/'+current_time

    headers = ["EPOCHS","BATCH_SIZE", "VAL_SPLIT", "COMPILER", "LEARNING_RATE", "NUM_NEURONS", "LOSS", "SEED"]
    df_parameters = pd.DataFrame([[EPOCHS, BATCH_SIZE, VALIDATION_SPLIT, COMPILER, LEARNING_RATE, NUM_NEURONS, LOSS, SEED]], columns=headers)

    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
        df_parameters.to_csv(log_dir+"/info_parameters.csv", index=False)

    # Path for image
    img_file_path = '../../figures/saliency_lstm_train_loss_class_NEWCRIS.png'


    def data_to_input(df_embeddings, df_ground_truth):
        i = 0
        df_input = np.zeros((len(df_ground_truth), SEQUENCE_LENGTH, EMBEDDING_LENGTH))
        for video_name in df_ground_truth['video'].apply(lambda x : x.split('.')[0]).values:
            video_embeddings = df_embeddings.loc[df_embeddings['0'] == video_name]
            video_embeddings = video_embeddings.iloc[:,3:].to_numpy()
            df_input[i] = video_embeddings
            i += 1

        return df_input

    def tf_pearson(y_true, y_pred):
        return tf.contrib.metrics.streaming_pearson_correlation(y_pred, y_true)[1]

    def step_decay(epoch):
       initial_lrate = LEARNING_RATE
       drop = 0.5
       epochs_drop = 25#settings.epochs_dr #10.0
       lrate = initial_lrate * math.pow(drop,
               math.floor((1+epoch)/epochs_drop))
       print("new lr:", str(lrate))
       return lrate


    def to_classifier(x) :
        if x > 0.77:
            return 1
        return 0

    #check normalization
    if(not NORMALIZATION):
        if(os.path.isfile(save_processed_files+"X_train.npy")):
            #load data
            X_train = np.load(save_processed_files+"X_train.npy")
            y_train = np.load(save_processed_files + "y_train.npy")
            X_test = np.load(save_processed_files + "X_test.npy")
            y_test = np.load(save_processed_files + "y_test.npy")
        else:
            # Load dataframes
            df_train_embeddings = pd.read_csv(train_embeddings_file_path)
            df_test_embeddings = pd.read_csv(test_embeddings_file_path)
            df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
            df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

            # Sort dataframes
            df_train_embeddings = df_train_embeddings.sort_values(['0', '1'], ascending=[True, True])
            df_test_embeddings = df_test_embeddings.sort_values(['0', '1'], ascending=[True, True])
            df_train_ground_truth = df_train_ground_truth.sort_values('video', ascending=True)
            df_test_ground_truth = df_test_ground_truth.sort_values('video', ascending=True)

            # Get X_train, Y_train, X_test, Y_test
            X_train = data_to_input(df_train_embeddings, df_train_ground_truth)
            X_test = data_to_input(df_test_embeddings, df_test_ground_truth)

            y_train = df_train_ground_truth['long-term_memorability'].to_numpy()
            y_test = df_test_ground_truth['long-term_memorability'].to_numpy()

            # SAVE FILES
            np.save(save_processed_files + "X_train.npy", X_train)
            np.save(save_processed_files + "y_train.npy", y_train)
            np.save(save_processed_files + "X_test.npy", X_test)
            np.save(save_processed_files + "y_test.npy", y_test)
    else:
        if (os.path.isfile(save_processed_files + "X_train_norm.npy")):
            # load data
            X_train = np.load(save_processed_files + "X_train_norm.npy")
            y_train = np.load(save_processed_files + "y_train.npy")
            X_test = np.load(save_processed_files + "X_test_norm.npy")
            y_test = np.load(save_processed_files + "y_test.npy")
        else:
            # Load dataframes
            df_train_embeddings = pd.read_csv(train_embeddings_file_path)
            df_test_embeddings = pd.read_csv(test_embeddings_file_path)
            df_train_ground_truth = pd.read_csv(train_ground_truth_file_path)
            df_test_ground_truth = pd.read_csv(test_ground_truth_file_path)

            # Sort dataframes
            df_train_embeddings = df_train_embeddings.sort_values(['0', '1'], ascending=[True, True])
            df_test_embeddings = df_test_embeddings.sort_values(['0', '1'], ascending=[True, True])
            df_train_ground_truth = df_train_ground_truth.sort_values('video', ascending=True)
            df_test_ground_truth = df_test_ground_truth.sort_values('video', ascending=True)

            #NORMALIZATION:
            columns2remove_train = df_train_embeddings.iloc[:,0:3]
            columns2remove_test = df_test_embeddings.iloc[:, 0:3]

            norm_scale = preprocessing.MinMaxScaler().fit(df_train_embeddings.iloc[:, 3:])
            new_df_embeddings_train = norm_scale.transform(df_train_embeddings.iloc[:, 3:])
            new_df_embeddings_test = norm_scale.transform(df_test_embeddings.iloc[:, 3:])

            df_train_embeddings = pd.concat([columns2remove_train, pd.DataFrame(new_df_embeddings_train)], axis=1)
            df_test_embeddings = pd.concat([columns2remove_test, pd.DataFrame(new_df_embeddings_test)], axis=1)

            # Get X_train, Y_train, X_test, Y_test
            X_train = data_to_input(df_train_embeddings, df_train_ground_truth)
            X_test = data_to_input(df_test_embeddings, df_test_ground_truth)

            y_train = df_train_ground_truth['long-term_memorability'].to_numpy()
            y_test = df_test_ground_truth['long-term_memorability'].to_numpy()

            # SAVE FILES
            np.save(save_processed_files + "X_train_norm.npy", X_train)
            np.save(save_processed_files + "y_train.npy", y_train)
            np.save(save_processed_files + "X_test_norm.npy", X_test)
            np.save(save_processed_files + "y_test.npy", y_test)

    if(classification):
        y_train_new = []
        y_test_new = []
        for label in y_train:
            y_train_new += [to_classifier(label)]
        for label in y_test:
            y_test_new += [to_classifier(label)]
        y_train = np.asarray(y_train_new)
        y_test = np.asarray(y_test_new)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y_train.reshape(-1, 1))
        y_train_oneHot = enc.transform(y_train.reshape(-1, 1)).toarray()
        y_test_oneHot = enc.transform(y_test.reshape(-1, 1)).toarray()
        NEURONS_LAST_LAYER = 2
        ACTIVATION_LAST_LAYER = 'softmax'
    else:
        y_test_oneHot = y_test
        y_train_oneHot = y_train
        NEURONS_LAST_LAYER = 1
        ACTIVATION_LAST_LAYER = 'sigmoid'


    print (X_train[0,:,:])
    print (y_train[0])
    print (X_train.shape)
    print(X_test.shape)
    print (y_train.shape)
    print(y_test.shape)


    # Now define the model
    #initial_input = Input(shape=(SEQUENCE_LENGTH,EMBEDDING_LENGTH))
    input_layer = keras.layers.Input(shape=(SEQUENCE_LENGTH,EMBEDDING_LENGTH),name="input")
    x = LSTM(units=NUM_NEURONS, return_sequences=False, name="LSTM_0")(input_layer) #dropout=0, recurrent_dropout=0.0,
    output = Dense(NEURONS_LAST_LAYER, activation=ACTIVATION_LAST_LAYER, name="output_layer")(x)
    model = keras.models.Model([input_layer], output)
    model.summary()


    #CALLBACKS
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',#val_loss
                                                  patience=40, verbose=2, mode='auto',
                                                  restore_best_weights=True)  # EARLY STOPPING

    #lrate_callback = keras.callbacks.LearningRateScheduler(step_decay)#DECAY LEARNING RATE #más info sobre learning rate decay methods in: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
    tensorboard = TensorBoard(log_dir=log_dir,update_freq='epoch',write_images=True,write_graph=False) #CONTROL THE TRAINING

    optim, lossFunct = nw_config.get_optim_and_lossFunct(optimizer = COMPILER, learning_rate = LEARNING_RATE, lossFunction=LOSS)
    model.compile(optimizer=optim, loss=lossFunct, metrics=METRICS)
    history = model.fit(X_train, y_train_oneHot,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT,callbacks=[tensorboard,earlyStopping])#lrate_callback
    #in order to see the info plotted by the tensorboard callback, open a terminal and type: tensorboard --logdir=/....Ruta aquí.../tfg-memorability-challenge/models/LOGS_saliencyLOGS_saliency/


    #SAVE RESULTS & PLOTS
    if(classification):
        print("done")
        score_test, accuracy_test = model.evaluate(X_test, y_test_oneHot,
                                                                 batch_size=BATCH_SIZE)

        score_train, accuracy_train = model.evaluate(X_train, y_train_oneHot,
                                                                     batch_size=BATCH_SIZE)
        print('Test score with LSTM:', score_test)
        print('Test acc with LSTM:', accuracy_test)

        headers_results = ["score_train", "acc_train", "score_test", "acc_test"]
        df_results = pd.DataFrame([[score_train, accuracy_train, score_test, accuracy_test]],columns=headers_results)
        #obtain predictions:
        pred_test = model.predict(X_test,batch_size=X_train.shape[0])
        pred_test = pred_test.reshape(-1)
        true = y_test_oneHot.reshape(-1)

    else:

        score_test, mse_test, CC_test, CCC_test= model.evaluate(X_test, y_test_oneHot,
                                    batch_size=X_test.shape[0])

        score_train, mse_train, CC_train, CCC_train = model.evaluate(X_train, y_train_oneHot,
                                    batch_size=X_train.shape[0])

        headers_results = ["score_train","mse_train", "CC_train", "CCC_train", "score_test","mse_test", "CC_test", "CCC_test"]
        df_results = pd.DataFrame([[score_train,mse_train, CC_train, CCC_train, score_test,mse_test, CC_test, CCC_test]], columns=headers_results)
        # obtain predictions: ----CHECKING THAT KERAS METHODS WORK-----------
        pred_test = model.predict(X_test, batch_size=X_test.shape[0])
        pred_test = pred_test.reshape(-1)
        true = y_test_oneHot.reshape(-1)
        CC_test = metrics_regression.PC_numpy(y_true=true, y_pred=pred_test)
        CCC_test = metrics_regression.CCC(y_true=true, y_pred=pred_test)
        CC_spearman_test = spearmanr(true,pred_test)[0]
        print("\n")
        print('Test score with LSTM:', score_test)
        print('Test mse with LSTM:', mse_test)
        print('Test CC with LSTM:', CC_test)
        print('Test CC SPEARMAN with LSTM:', CC_spearman_test)
        print('Test CCC with LSTM:', CCC_test)


        pred_train = model.predict(X_train, batch_size=X_train.shape[0])
        pred_train = pred_train.reshape(-1)
        true = y_train_oneHot.reshape(-1)
        CC_train = metrics_regression.PC_numpy(y_true=true, y_pred=pred_train)
        CCC_train = metrics_regression.CCC(y_true=true, y_pred=pred_train)
        CC_spearman_train = spearmanr(true, pred_train)[0]
        print("\n")
        print('train score with LSTM:', score_train)
        print('train mse with LSTM:', mse_train)
        print('train CC with LSTM:', CC_train)
        print('train CC SPEARMAN with LSTM:', CC_spearman_train)
        print('train CCC with LSTM:', CCC_train)

    df_results.to_csv(log_dir+"/info_results.csv", index=False)


    with open(model_save_path, 'w+') as save_file:
        save_file.write(model.to_json())

    model.save_weights(weight_save_path)

    # Plotting loss
    plt.suptitle('Optimizer : adagrad', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.legend(loc='upper right')

    # Save img
    plt.savefig(img_file_path)

    plt.show()
