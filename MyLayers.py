import keras
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils.np_utils import *
from keras import backend as K

# Input shape = [(None, timesteps, input_dim), (None, timesteps, input_dim)]
# 0 ==> input
# 1 ==> attention

class Attention_GRU(Layer):
    '''
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use
        (see [activations](../activations.md)).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
        for the recurrent step
        (see [activations](../activations.md)).
    use_bias: Boolean, whether the layer uses a bias vector.
    return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence.
    go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
    input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
            input_length: Length of input sequences, to be specified
            when it is constant.
            This argument is required if you are going to connect
            `Flatten` then `Dense` layers upstream
            (without it, the shape of the dense outputs cannot be computed).
            Note that if the recurrent layer is not the first layer
            in your model, you would need to specify the input length
            at the level of the first layer
            (e.g. via the `input_shape` argument)
    '''

    def __init__(self, units,
                 return_sequences=True,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 implementation=0,

                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,

                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(Attention_GRU, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll
        self.implementation = implementation
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        self.state_spec = None
        self.dropout = 0
        self.recurrent_dropout = 0

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('An Attention_GRU layer takes input as [input_seq, att_seq]')

        batch_size = input_shape[0][0]
        self.input_dim = input_shape[0][2]
        self.att_dim = input_shape[1][2]
        self.input_spec = [InputSpec(shape=(batch_size, None, self.input_dim)), InputSpec(shape=(batch_size, None, self.att_dim))]
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]

        self.kernel = self.add_weight((self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            (self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units * 3,),
                                        name='bias',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        self.built = True

    def call(self, inputs, mask=None, initial_state=None, training=None):
        constants = self.get_constants(inputs, training=None)           # [dp_mask, rec_dp_mask, att]
        preprocessed_input = self.preprocess_input(inputs, training=None)
        input_shape = K.int_shape(inputs[0])

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             self.get_initial_states(inputs),
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            return outputs
        else:
            return last_output

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs[0])
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_z = recurrent._time_distributed_dense(inputs[0], self.kernel_z, self.bias_z,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = recurrent._time_distributed_dense(inputs[0], self.kernel_r, self.bias_r,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_h = recurrent._time_distributed_dense(inputs[0], self.kernel_h, self.bias_h,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            # return inputs
            raise NotImplementedError

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation != 0 and 0 < self.dropout < 1:
            raise NotImplementedError
            # input_shape = K.int_shape(inputs)
            # input_dim = input_shape[-1]
            # ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            # ones = K.tile(ones, (1, int(input_dim)))

            # def dropped_inputs():
            #     return K.dropout(ones, self.dropout)

            # dp_mask = [K.in_train_phase(dropped_inputs,
            #                             ones,
            #                             training=training) for _ in range(3)]
            # constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[0][:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        constants.append(inputs[1])
        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs[0])  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def step(self, inputs, states):
        h_tm1 = states[0]  # previous memory
        dp_mask = states[1]  # dropout matrices for recurrent units
        rec_dp_mask = states[2]
        att_seq = states[3] # attention sequence

        # if self.implementation == 2:
        #     matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
        #     if self.use_bias:
        #         matrix_x = K.bias_add(matrix_x, self.bias)
        #     matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
        #                          self.recurrent_kernel[:, :2 * self.units])

        #     x_z = matrix_x[:, :self.units]
        #     x_r = matrix_x[:, self.units: 2 * self.units]
        #     recurrent_z = matrix_inner[:, :self.units]
        #     recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        #     z = self.recurrent_activation(x_z + recurrent_z)
        #     r = self.recurrent_activation(x_r + recurrent_r)

        #     x_h = matrix_x[:, 2 * self.units:]
        #     recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
        #                         self.recurrent_kernel[:, 2 * self.units:])
        #     hh = self.activation(x_h + recurrent_h)
        # else:
        #     if self.implementation == 0:
        #         x_z = inputs[:, :self.units]
        #         x_r = inputs[:, self.units: 2 * self.units]
        #         x_h = inputs[:, 2 * self.units:]
        #     elif self.implementation == 1:
        #         x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
        #         x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
        #         x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
        #         if self.use_bias:
        #             x_z = K.bias_add(x_z, self.bias_z)
        #             x_r = K.bias_add(x_r, self.bias_r)
        #             x_h = K.bias_add(x_h, self.bias_h)
        #     else:
        #         raise ValueError('Unknown `implementation` mode.')
        #     z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
        #                                               self.recurrent_kernel_z))
        #     r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
        #                                               self.recurrent_kernel_r))

        #     hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
        #                                      self.recurrent_kernel_h))

        if self.implementation != 0:
            raise NotImplementedError
        x_z = inputs[:, :self.units]
        x_r = inputs[:, self.units: 2 * self.units]
        x_h = inputs[:, 2 * self.units:]

        z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
                                                  self.recurrent_kernel_z))
        r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
                                                  self.recurrent_kernel_r))

        hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
                                         self.recurrent_kernel_h))


        h = z * h_tm1 + (1 - z) * hh

        # h = (batches, dim)
        # att_seq = (batched, timesteps, dim)
        a = K.batch_dot(h, att_seq, axes=[1,2])   # a = (batch, timesteps)
        o = K.batch_dot(att_seq, a, axes=[1,1])

        if 0 < self.dropout + self.recurrent_dropout:
            o._uses_learning_phase = True
        return o, [o]

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'implementation': self.implementation,
                  'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(Attention_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


