import keras
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils.np_utils import *
from keras.activations import softmax
from keras import backend as K

# Input shape = [(None, timesteps, input_dim), (None, timesteps, input_dim)]
# 0 ==> input (former seq)
# 1 ==> attention (encoder hidden states)

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

        # W_xz, W_xr, W_xh
        self.W_xA = self.add_weight((self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # W_hz, W_hr, W_hh
        self.W_hA = self.add_weight(
            (self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # W_cz, W_cr, W_ch
        self.W_cA = self.add_weight(
            (self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units * 3,),
                                        name='bias',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None


        self.W_xz = self.W_xA[:, :self.units]
        self.W_hz = self.W_hA[:, :self.units]
        self.W_cz = self.W_cA[:, :self.units]
        self.W_xr = self.W_xA[:, self.units: self.units * 2]
        self.W_hr = self.W_hA[:, self.units: self.units * 2]
        self.W_cr = self.W_cA[:, self.units: self.units * 2]
        self.W_xh = self.W_xA[:, self.units * 2:]
        self.W_hh = self.W_hA[:, self.units * 2:]
        self.W_ch = self.W_cA[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        # Added parameters
        # self.after_att_layers_cnt = 3    # You can tune this

        # self.after_att_kernel = self.add_weight((self.units, self.units * self.after_att_layers_cnt),
        #                                 name='after_att_kernel',
        #                                 initializer=self.kernel_initializer,
        #                                 regularizer=self.kernel_regularizer,
        #                                 constraint=self.kernel_constraint)

        # self.after_att_kernel_list = [self.after_att_kernel[:, self.units * i: self.units * (i + 1)] for i in range(self.after_att_layers_cnt)]

        # self.after_att_bias = self.add_weight((self.units * self.after_att_layers_cnt,),
        #                                 name='after_att_bias',
        #                                 initializer='zero',
        #                                 regularizer=self.bias_regularizer,
        #                                 constraint=self.bias_constraint)

        # self.after_att_bias_list = [self.after_att_bias[self.units * i: self.units * (i + 1)] for i in range(self.after_att_layers_cnt)]

        self.built = True

    def call(self, inputs, mask=None, initial_state=None, training=None):
        constants = self.get_constants(inputs, training=None)           # [dp_mask, rec_dp_mask, att]
        preprocessed_input = self.preprocess_input(inputs, training=None)
        input_shape = K.int_shape(inputs[0])

        if initial_state == None:
            initial_state = self.get_initial_states(inputs)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
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

            x_z = recurrent._time_distributed_dense(inputs[0], self.W_xz, self.bias_z,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = recurrent._time_distributed_dense(inputs[0], self.W_xr, self.bias_r,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_h = recurrent._time_distributed_dense(inputs[0], self.W_xh, self.bias_h,
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

        # h_tm1 = (batches, dim)
        # att_seq = (batched, timesteps, dim)
        a = K.batch_dot(h_tm1, att_seq, axes=[1,2])   # a = (batch, timesteps)
        a = K.softmax(a)
        c = K.batch_dot(att_seq, a, axes=[1,1])

        x_z = inputs[:, :self.units]
        x_r = inputs[:, self.units: 2 * self.units]
        x_h = inputs[:, 2 * self.units:]

        z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0], self.W_hz) + K.dot(c, self.W_cz))
        r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1], self.W_hr) + K.dot(c, self.W_cr))
        h_ = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2], self.W_hh) + K.dot(c, self.W_ch))

        h = z * h_tm1 + (1 - z) * h_

        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'implementation': self.implementation,
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


# Input shape = [(None, timesteps, input_dim), (None, sens, input_dim), (None, sens, timesteps, input_dim)]
# 0 ==> input (former seq)
# 1 ==> hidden states for sens (high layer)
# 2 ==> hiddem states for words (low layer)

class Attention_2H_GRU(Layer):
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
        super(Attention_2H_GRU, self).__init__(**kwargs)
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
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3), InputSpec(ndim=4)]
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
        return None

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('An Attention_GRU layer takes input as [input_seq, att_sens, att_words]')

        batch_size = input_shape[0][0]
        self.input_dim = input_shape[0][2]
        self.sens = input_shape[1][1]
        self.att_s_dim = input_shape[1][2]      # Attention for sens
        self.att_w_dim = input_shape[2][3]      # Attention for words
        self.input_spec = [InputSpec(shape=(batch_size, None, self.input_dim)), InputSpec(shape=(batch_size, self.sens, self.att_s_dim)), InputSpec(shape=(batch_size, self.sens, None, self.att_w_dim))]
        self.state_spec = InputSpec(shape=(batch_size, self.units))

        self.states = [None]

        # W_xz, W_xr, W_xh
        self.W_xA = self.add_weight((self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # W_hz, W_hr, W_hh
        self.W_hA = self.add_weight(
            (self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # W_cz, W_cr, W_ch
        self.W_cA = self.add_weight(
            (self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units * 3,),
                                        name='bias',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None


        self.W_xz = self.W_xA[:, :self.units]
        self.W_hz = self.W_hA[:, :self.units]
        self.W_cz = self.W_cA[:, :self.units]
        self.W_xr = self.W_xA[:, self.units: self.units * 2]
        self.W_hr = self.W_hA[:, self.units: self.units * 2]
        self.W_cr = self.W_cA[:, self.units: self.units * 2]
        self.W_xh = self.W_xA[:, self.units * 2:]
        self.W_hh = self.W_hA[:, self.units * 2:]
        self.W_ch = self.W_cA[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None

        # Added parameters
        self.att_layers_cnt = 3    # You can tune this

        self.att_h1_kernel_list = []
        self.att_h1_kernel_list.append(self.add_weight((self.units + self.att_w_dim, self.units),
                                        name='att_h1_kernel_1',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))
        self.att_h1_kernel_list.append(self.add_weight((self.units, self.units),
                                        name='att_h1_kernel_2',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))
        self.att_h1_kernel_list.append(self.add_weight((self.units, 1),
                                        name='att_h1_kernel_3',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))

        self.att_h1_bias_list = []
        self.att_h1_bias_list.append(self.add_weight((self.units,),
                                        name='att_h1_bias_1',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))
        self.att_h1_bias_list.append(self.add_weight((self.units,),
                                        name='att_h1_bias_2',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))
        self.att_h1_bias_list.append(self.add_weight((1,),
                                        name='att_h1_bias_3',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))

        self.att_h2_kernel_list = []
        self.att_h2_kernel_list.append(self.add_weight((self.units + self.att_s_dim, self.units),
                                        name='att_h2_kernel_1',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))
        self.att_h2_kernel_list.append(self.add_weight((self.units, self.units),
                                        name='att_h2_kernel_2',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))
        self.att_h2_kernel_list.append(self.add_weight((self.units, 1),
                                        name='att_h2_kernel_3',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint))

        self.att_h2_bias_list = []
        self.att_h2_bias_list.append(self.add_weight((self.units,),
                                        name='att_h2_bias_1',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))
        self.att_h2_bias_list.append(self.add_weight((self.units,),
                                        name='att_h2_bias_2',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))
        self.att_h2_bias_list.append(self.add_weight((1,),
                                        name='att_h2_bias_3',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint))

        self.built = True

    def call(self, inputs, mask=None, initial_state=None, training=None):
        constants = self.get_constants(inputs, mask, training=None)         # [dp_mask, rec_dp_mask, att_sens, att_words, att_words_mask]
        preprocessed_input = self.preprocess_input(inputs, training=None)
        input_shape = K.int_shape(inputs[0])

        if initial_state == None:
            initial_state = self.get_initial_states(inputs)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
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

            x_z = recurrent._time_distributed_dense(inputs[0], self.W_xz, self.bias_z,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = recurrent._time_distributed_dense(inputs[0], self.W_xr, self.bias_r,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_h = recurrent._time_distributed_dense(inputs[0], self.W_xh, self.bias_h,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            return K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            # return inputs
            raise NotImplementedError

    def get_constants(self, inputs, mask, training=None):
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

        constants += [inputs[1], inputs[2], mask[2]]
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
        h_tm1 = states[0]       # previous memory
        dp_mask = states[1]     # dropout matrices for recurrent units
        rec_dp_mask = states[2]
        att_sens = states[3]    # hidden states for sens (high layer)
        att_words = states[4]   # hiddem states for words (low layer)
        att_words_mask = states[5]  # mask for 'att_words'
        # h_tm1 = (batch, dim)
        # att_sens = (batch, sens, dim)
        # att_words = (batch, sens, timesteps, dim)


        # _shp = K.int_shape(att_words)                       # _shp = tuple(batch, sens, timesteps, dim)
        # _h_tm1 = K.expand_dims(h_tm1, axis=1)               # h_tm1 = (batch, 1, dim)

        # alpha = K.batch_dot(att_sens, _h_tm1, axes=[2,2])   # alpha = (batch, sens, 1)
        # alpha = softmax(alpha, axis=1)
        # alpha = K.expand_dims(alpha)                        # alpha = (batch, sens, 1, 1)

        # _aw = K.reshape(att_words, shape=(-1, _shp[1] * _shp[2], _shp[3]))         # _aw = (batch, sens * timesteps, dim)
        # beta = K.batch_dot(_aw, _h_tm1, axes=[2,2])             # beta = (batch, sens * timesteps, 1)
        # beta = K.reshape(beta, shape=(-1, _shp[1], _shp[2]))    # beta = (batch, sens, timesteps)
        # beta = K.exp(beta) * K.cast(att_words_mask, dtype='float32')
        # beta = beta / K.sum(beta, axis=-1, keepdims=True)
        # beta = K.expand_dims(beta)                          # beta = (batch, sens, timesteps, 1)
        # c = alpha * beta * att_words                        # c = (batch, sens, timesteps, dim)
        # c = K.sum(K.sum(c, axis=2), axis=1)                 # c = (batch, dim)

        _shp = K.int_shape(att_words)                       # _shp = tuple(batch, sens, timesteps, dim)
        _h_tm1 = K.expand_dims(h_tm1, axis=1)               # _h_tm1 = (batch, 1, dim)
        _h_tm1 = K.tile(_h_tm1, (1, _shp[1], 1))             # _h_tm1 = (batch, sens, dim)

        alpha = K.concatenate([_h_tm1, att_sens])           # alpha = (batch, sens, 2 * dim)
        for i in range(self.att_layers_cnt):
            alpha = K.relu(K.bias_add(K.dot(alpha, self.att_h2_kernel_list[i]), self.att_h2_bias_list[i]))
                                                            # alpha = (batch, sens, 1)
        alpha = K.expand_dims(alpha)                        # alpha = (batch, sens, 1, 1)


        _h_tm1 = K.expand_dims(_h_tm1, axis=2)
        _h_tm1 = K.tile(_h_tm1, (1, 1, _shp[2], 1))
        beta = K.concatenate([_h_tm1, att_words])
        for i in range(self.att_layers_cnt):
            beta = K.relu(K.bias_add(K.dot(beta, self.att_h1_kernel_list[i]), self.att_h1_bias_list[i]))
                                                            # beta = (batch, sens, timesteps, 1)
        c = alpha * beta * att_words                        # c = (batch, sens, timesteps, dim)
        c = K.sum(K.sum(c, axis=2), axis=1)                 # c = (batch, dim)

        x_z = inputs[:, :self.units]
        x_r = inputs[:, self.units: 2 * self.units]
        x_h = inputs[:, 2 * self.units:]

        z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0], self.W_hz) + K.dot(c, self.W_cz))
        r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1], self.W_hr) + K.dot(c, self.W_cr))
        h_ = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2], self.W_hh) + K.dot(c, self.W_ch))

        h = z * h_tm1 + (1 - z) * h_

        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'implementation': self.implementation,
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
        base_config = super(Attention_2H_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Input shape = [(None, timesteps, input_dim), (None, timesteps, input_dim)]
# 0 ==> input
# 1 ==> attention
# Not available now (old implementation of attention)

class Attention_Feedback_GRU(Layer):
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
                 id2v,
                 hard_argmax=False,
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
        super(Attention_Feedback_GRU, self).__init__(**kwargs)
        self.units = units
        self.id2v = id2v
        self.hard_argmax = hard_argmax
        self.vocab_size = id2v.shape[0]
        self.wv_dim = id2v.shape[1]
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

        # Parameters for attention
        self.after_att_layers_cnt = 2    # You can tune this

        self.after_att_kernel = self.add_weight((self.units, self.units * self.after_att_layers_cnt),
                                        name='after_att_kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        self.after_att_kernel_list = [self.after_att_kernel[:, self.units * i: self.units * (i + 1)] for i in range(self.after_att_layers_cnt)]

        self.after_att_bias = self.add_weight((self.units * self.after_att_layers_cnt,),
                                        name='after_att_bias',
                                        initializer='zero',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        self.after_att_bias_list = [self.after_att_bias[self.units * i: self.units * (i + 1)] for i in range(self.after_att_layers_cnt)]

        # Parameters for feedback
        self.W_o1 = self.add_weight((self.units, self.vocab_size),
                                        name='W_o1',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.embedding = self.add_weight((self.vocab_size, self.wv_dim),
                                        name='embedding',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=False)
        K.set_value(self.embedding, self.id2v)
        self.built = True

    def call(self, inputs, mask=None, initial_state=None, training=None):
        constants = self.get_constants(inputs, training=None)           # [dp_mask, rec_dp_mask, att]
        preprocessed_input = self.preprocess_input(inputs, training=None)
        input_shape = K.int_shape(inputs[0])

        if initial_state == None:
            initial_state = self.get_initial_states(inputs)

        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_state,
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

        # h = (batch, dim)
        # att_seq = (batch, timesteps, dim)
        a = K.batch_dot(h, att_seq, axes=[1,2])     # a = (batch, timesteps)
        a = K.softmax(a)
        o = K.batch_dot(att_seq, a, axes=[1,1])     # o = (batch, dim)

        for i in range(self.after_att_layers_cnt):
            o = K.relu(K.dot(o, self.after_att_kernel_list[i]) + self.after_att_bias_list[i], alpha=0.01)

        # W_o1 = (dim, vocab_size)
        d = K.softmax(K.dot(o, self.W_o1))   # d = (batch, vocab_size)

        if self.hard_argmax:
            w_id = K.argmax(d)
            w_v = self.embedding[w_id]
        else:
            w_v = K.dot(d, self.embedding)

        if 0 < self.dropout + self.recurrent_dropout:
            w_v._uses_learning_phase = True
        return d, [w_v]

    def get_config(self):
        config = {'units': self.units,
                  'id2v' : self.id2v,
                  'hard_argmax' : self.hard_argmax,
                  'return_sequences': self.return_sequences,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll,
                  'implementation': self.implementation,
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
        base_config = super(Attention_Feedback_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

