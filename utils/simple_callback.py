
import tensorflow as tf
import keras

#CALLBACKS
# Function to display the target and prediciton
class simple_Callback(keras.callbacks.Callback):
    def __init__(self,
                 output_layer_tensors_name=None):
        self.metrics_name = output_layer_tensors_name
        super(simple_Callback, self).__init__()


    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []


    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        # for output_layer_i in self.metrics_name:
        #     print("----------------------"+output_layer_i+"(below)-----------------------------")
        #     print(logs[output_layer_i])
        return