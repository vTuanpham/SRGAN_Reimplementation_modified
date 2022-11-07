import keras.backend as K
from keras.layers import Layer

#Custom layer Inherited from keras.layers.Layer
class SymmetricPadding2D(Layer):

    #Initialize
    def __init__(self, output_dim=1,name=None,padding=[1,1],data_format="channels_last", **kwargs):
        super(SymmetricPadding2D,self).__init__(name=name)
        self.output_dim = output_dim
        self.data_format = data_format
        self.padding = padding
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SymmetricPadding2D, self).build(input_shape)

    #Since the layer are not learning anything, just padding, we don't need to have config.update implement
    def get_config(self):
        config = super(SymmetricPadding2D,self).get_config()
        return config

    #Function call
    def call(self, inputs):
        #Padding order depend on dataformat
        if self.data_format == "channels_last":
            #(batch, depth, rows, cols, channels)
            pad = [[0,0]] + [[i,i] for i in self.padding] + [[0,0]]
        elif self.data_format == "channels_first":
            #(batch, channels, depth, rows, cols)
            pad = [[0, 0], [0, 0]] + [[i,i] for i in self.padding]

        if K.backend() == "tensorflow":
            import tensorflow as tf
            paddings = tf.constant(pad)
            out = tf.pad(inputs, paddings, "REFLECT")
        else:
            raise Exception("Backend " + K.backend() + "not implemented")
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)