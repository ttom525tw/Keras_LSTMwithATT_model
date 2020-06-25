# -*- coding: utf-8 -*-
from keras.engine.topology import Layer,InputSpec
from keras import backend as  initializers,regularizers,activations,constraints
import keras
class SelfAttention(Layer):
    def __init__(self,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SelfAttention, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        time_steps = input_shape[1]
        dimensions = input_shape[1]

        self.attention = keras.models.Sequential(name='attention')
        # starting off, each element of the batch is (time_steps, dimensions)
        # turn this into (time_steps, 1)

        # attention matrix, this is the main thing being learned
        self.attention.add(keras.layers.Dense(dimensions,
                                              input_shape=(
                                                  time_steps, dimensions,),
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        self.attention.add(keras.layers.Activation(self.activation))
        # now convert to an attention vector
        self.attention.add(keras.layers.Dense(1,
                                              kernel_initializer=self.kernel_initializer,
                                              kernel_regularizer=self.kernel_regularizer,
                                              kernel_constraint=self.kernel_constraint))
        # make an attention vector
        self.attention.add(keras.layers.Flatten())
        self.attention.add(keras.layers.Activation('softmax'))
        # repeat this time step weighting for each dimensions
        self.attention.add(keras.layers.RepeatVector(dimensions))
        # reshape to be (time_steps, dimensions)
        self.attention.add(keras.layers.Permute([2, 1]))

        # not using add_weight, so update the weighs manually
        self.trainable_weights = self.attention.trainable_weights
        self.non_trainable_weights = self.attention.non_trainable_weights

        # all done
        self.built = True

    def call(self, inputs):
        # build the attention matrix
        attention = self.attention(inputs)
        # apply the attention matrix with element wise multiplication
        return keras.layers.Multiply()([inputs, attention])

    def compute_output_shape(self, input_shape):
        # there is no change in shape, the values are just weighted
        return input_shape

    def get_config(self):
        config = {
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        return dict(config)