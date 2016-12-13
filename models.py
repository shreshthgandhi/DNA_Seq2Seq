import tensorflow as tf
import numpy as np
import DNA_reader as reader
import time

class SmallConfig(object):
    """Small Configuration"""
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 2
    num_steps_encoder = 10
    num_steps_decoder = 15
    batch_size = 10
    hidden_size = 200
    max_epochs = 2
    max_max_epochs = 6
    lr_decay = 0.5
    init_scale = 0.1
    vocab_size = 8 # 3 extra tokens PAD,GO and EOS and 0 is unused
    compression_dims = 2
    teacher_forcing = False
    checkpoint_dir = 'checkpoint'

def get_config(flag='small'):
    if flag=='small':
        return SmallConfig()


def data_type():
    return tf.float32


class DNA_input(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps_encoder = num_steps_encoder = config.num_steps_encoder
        self.num_steps_decoder = num_steps_decoder = config.num_steps_decoder
        self.epoch_size = len(data[0]) // batch_size
        (self.encoder_input, self.decoder_input,
         self.decoder_targets, self.labels) = reader.DNA_producer(data, config, name)


class DNA_seq_model(object):
    def __init__(self, is_training, config, input_):
        self._input = input_
        self.batch_size = input_.batch_size
        self.num_steps_encoder = input_.num_steps_encoder
        self.num_steps_decoder = input_.num_steps_decoder
        self.num_layers = config.num_layers
        self.size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.teacher_force = config.teacher_forcing
        self.compression_dims = config.compression_dims
        self.checkpoint_dir = config.checkpoint_dir
        self.vocab_size = config.vocab_size

    def build_model(self):
        #         weight_initializer = tf.random_normal_initializer(mean = 0.0, stddev=1.0)
        self.encoder_input = tf.placeholder(tf.int32, [None, self.num_steps_encoder])
        self.decoder_input = tf.placeholder(tf.int32,[None, self.num_steps_decoder])
        self.decoder_targets = tf.placeholder(tf.int32,[None, self.num_steps_decoder])
        self.labels = tf.placeholder(tf.int32, [None])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
        self.initial_state = cell.zero_state(self.batch_size, data_type())
        temp_state = []
        with tf.variable_scope("Encoder_initial_state"):
            for i, (c, h) in enumerate(self.initial_state):
                temp = self.initial_state[i]._replace(c=tf.get_variable("init_c" + str(i), [batch_size, size]),
                                                       h=tf.get_variable("init_h" + str(i), [batch_size, size]))
                temp_state.append(temp)
        self.initial_state = tuple(temp_state)
        temp_state = []
        with tf.variable_scope("Decoder_initial_state"):
            for i, (c, h) in enumerate(self.initial_state):
                temp = self.initial_state[i]._replace(c=tf.get_variable("init_c" + str(i), [batch_size, size]),
                                                       h=tf.get_variable("init_h" + str(i), [batch_size, size]))
                temp_state.append(temp)
        self.decoder_initial_state = tuple(temp_state)
        #             self._initial_state[i].c = tf.get_variable("init_c"+str(i), [batch_size, size])
        #             self._initial_state[i].h = tf.get_variable("init_h"+str(i), [batch_size, size])

        #         lstm_cell_decoder = tf.nn.rnn_cell.BasicLSTMCell(size + compression_dims, forget_bias=0.0, state_is_tuple=True)
        #         decoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_decoder] * num_layers, state_is_tuple=True)
        #         decoder_initial_state = decoder_cell.zero_state(batch_size, data_type())
        with tf.variable_scope('Decoder'):
            softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            self.encoder_inputs = tf.nn.embedding_lookup(embedding, self.encoder_input)
            self.decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_input)
        self.encoded_state = self.encoder(self.encoder_inputs)
        self.hidden_state = self.generate_hidden_state(self.encoded_state)
        self.decoder_outputs, self.decoder_state = self.decoder(self.hidden_state, self.decoder_inputs)



        # # encoder_outputs = []
        # # state = self._initial_state
        # # with tf.variable_scope("RNN_encoder"):
        # #     for time_step in range(num_steps_encoder):
        # #         if time_step > 0: tf.get_variable_scope().reuse_variables()
        # #         (cell_output, state) = cell(encoder_inputs[:, time_step, :], state)
        # #         encoder_outputs.append(cell_output)
        # #         #         with tf.variable_scope("state_transform"):
        # W_compress = tf.get_variable("W_compress", [2 * num_layers * size, compression_dims], dtype=data_type())
        # b_compress = tf.get_variable("bcompress", [compression_dims], dtype=data_type())
        # W_sigma = tf.get_variable("W_sigma", [2 * num_layers * size, compression_dims], dtype=data_type())
        # b_sigma = tf.get_variable("b_sigma", [compression_dims], dtype=data_type())
        #
        # #         W_expand = tf.get_variable("W_expand", [compression_dims,2*num_layers*size], dtype=data_type())
        # #         b_expand = tf.get_variable("b_expand", [2*num_layers*size], dtype=data_type())
        #
        # eps = tf.random_normal([compression_dims])
        #
        # self._encoded_state = state
        # self.compressed_state = tf.reshape(tf.transpose(tf.pack(state), [0, 1, 3, 2]), [-1, batch_size])
        # self.compressed_state = tf.transpose(self.compressed_state, [1, 0])
        #
        # mu = tf.matmul(self.compressed_state, W_compress) + b_compress
        # log_sigma_sq = tf.matmul(self.compressed_state, W_sigma) + b_sigma
        # #         variance = sigma*sigma
        #
        # self._hidden_state = hidden_state = mu + tf.sqrt(tf.exp(log_sigma_sq)) * eps

        #         self._hidden_state = hidden_state = tf.matmul(self.compressed_state, W_compress) + b_compress
        #         expanded_state = tf.matmul(hidden_state, W_expand) + b_expand
        #         expanded_state = tf.reshape(tf.transpose(expanded_state,[1,0]),[num_layers,2,size,-1])
        #         expanded_state = tf.transpose(expanded_state,[0,1,3,2])
        #         expanded_state = tf.unpack(expanded_state, axis=0)
        #         state_list = []
        #         for i,layer in enumerate(expanded_state):
        #             state_list.append(tuple(tf.unpack(layer, axis=0)))

        #         self.recovered_state = tuple(state_list)

        #         expanded_state= self.recovered_state



        #
        # hidden_state = tf.tile(hidden_state, [num_steps_decoder, 1])
        # hidden_state = tf.reshape(hidden_state, [-1, num_steps_decoder, compression_dims])
        # decoder_inputs = tf.concat(2, [decoder_inputs, hidden_state])
        # W_decoder = tf.get_variable("W_decoder", [size + compression_dims, size], dtype=data_type())
        # b_decoder = tf.get_variable("b_decoder", [size], dtype=data_type())
        # #         decoder_inputs = tf.batch_matmul(decoder_inputs, W_decoder)
        # decoder_inputs_list = tf.unpack(decoder_inputs, axis=1)
        # #         decoder_inputs_list =[]
        # for time_step in range(num_steps_decoder):
        #     #             if time_step > 0:tf.get_variable_scope().reuse_variables()
        #     decoder_inputs_list[time_step] = tf.matmul(decoder_inputs_list[time_step], W_decoder) + b_decoder


        #
        # (decoder_outputs, state) = tf.nn.seq2seq.rnn_decoder(decoder_inputs_list, self._decoder_initial_state, cell,
        #                                                      loop_function=loop if not (self._teacher_force) else None)
        # self._decoded_state = state



        self.final_state = tf.reshape(tf.transpose(tf.pack(state), [0, 1, 3, 2]), [-1, batch_size])
        self.final_state = tf.transpose(self.final_state, [1, 0])
        W_final = tf.get_variable("W_final", [2 * num_layers * size, 4])
        b_final = tf.get_variable("b_final", [4])
        self._final_label = tf.matmul(self.final_state, W_final) + b_final
        label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self._final_label, self._input.labels)
        self._label_loss = tf.reduce_sum(label_loss) / batch_size

        decoder_output = tf.reshape(tf.concat(1, decoder_outputs), [-1, size])

        logits = tf.matmul(decoder_output, softmax_w) + softmax_b
        self._probabilities = tf.reshape(tf.nn.softmax(logits), [batch_size, num_steps_decoder, vocab_size])

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.decoder_targets, [-1])],
            [tf.ones([batch_size * num_steps_decoder], dtype=data_type())])
        #         KL_loss = 0.5 * (tf.reduce_sum(variance,1) + tf.reduce_sum(mu * mu,1) - compression_dims +tf.log(tf.reduce_prod(variance,1)) )
        KL_loss = 0.5 * tf.reduce_sum(tf.exp(log_sigma_sq) + tf.square(mu) - log_sigma_sq - 1, 1)
        self._KL_loss = KL_loss_avg = tf.reduce_sum(KL_loss) / batch_size
        self._recon_loss = Reconstruction_loss_avg = tf.reduce_sum(loss) / batch_size
        self._anneal = tf.Variable(0.0, trainable=False)
        self._cost = cost = Reconstruction_loss_avg + self._anneal * KL_loss_avg

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        #         self._anneal = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        #         optimizer = tf.train.AdamOptimizer(self._lr)
        global_step = tf.contrib.framework.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=global_step)

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self._new_anneal = tf.placeholder(
            tf.float32, shape=[], name='new_anneal')
        self._anneal_update = tf.assign(self._anneal, self._new_anneal)

    def encoder(self, encoder_inputs):
        # encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
        # encoder_outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN_encoder"):
            for time_step in range(self.num_steps_encoder):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = self.cell(encoder_inputs[:, time_step, :], state)
                # encoder_outputs.append(cell_output)
        return state

    def decoder(self, hidden_state, decoder_inputs):

        vocab_size = self.vocab_size
        batch_size = self.batch_size
        hidden_state_expanded = tf.tile(hidden_state, [self.num_steps_decoder, 1])
        hidden_state_expanded = tf.reshape(hidden_state_expanded, [-1, self.num_steps_decoder, self.compression_dims])
        # decoder_inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
        decoder_inputs_expanded = tf.concat(2, [decoder_inputs, hidden_state_expanded])
        decoder_inputs_list = tf.unpack(decoder_inputs_expanded, axis=1)
        with tf.variable_scope('Decoder'):
            W_decoder = tf.get_variable("W_decoder", [size + compression_dims, size], dtype=data_type())
            b_decoder = tf.get_variable("b_decoder", [size], dtype=data_type())

        for time_step in range(self.num_steps_decoder):
            decoder_inputs_list[time_step] = tf.matmul(decoder_inputs_list[time_step], W_decoder) + b_decoder

        def loop(prev, _):
            #             prev = tf.matmul(prev, softmax_w) + softmax_b
            #             prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            #             true_symbol =  tf.nn.embedding_lookup(embedding, prev_symbol)
            GO_token = tf.constant(vocab_size - 3, shape=[batch_size])
            true_symbol = tf.nn.embedding_lookup(embedding, GO_token)  # For GO token
            appended_symbol = tf.concat(1, [true_symbol, self.hidden_state])
            final_symbol = tf.matmul(appended_symbol, W_decoder) + b_decoder
            return final_symbol

        (decoder_outputs, state) = tf.nn.seq2seq.rnn_decoder(decoder_inputs_list, self.decoder_initial_state, self.cell,
                                                             loop_function=loop if not (self.teacher_force) else None)
        return decoder_outputs, state

    def generate_hidden_state(self, encoded_state):
        with tf.variable_scope('Hidden_state'):
            W_compress = tf.get_variable("W_compress", [2 * self.num_layers * self.size, self.compression_dims], dtype=data_type())
            b_compress = tf.get_variable("bcompress", [self.compression_dims], dtype=data_type())
            W_sigma = tf.get_variable("W_sigma", [2 * self.num_layers * self.size, self.compression_dims], dtype=data_type())
            b_sigma = tf.get_variable("b_sigma", [self.compression_dims], dtype=data_type())
        eps = tf.random_normal([self.compression_dims])
        compressed_state = tf.reshape(tf.transpose(tf.pack(encoded_state), [0, 1, 3, 2]), [-1, self.batch_size])
        compressed_state = tf.transpose(compressed_state, [1, 0])
        self.mu = mu = tf.matmul(compressed_state, W_compress) +b_compress
        self.log_sigma_sq = log_sigma_sq =  tf.matmul(compressed_state, W_sigma) + b_sigma
        hidden_state = mu + tf.sqrt(tf.exp(log_sigma_sq))* eps
        return hidden_state

    def classifier(self, decoder_state):
        final_state =  tf.reshape(tf.transpose(tf.pack(decoder_state), [0, 1, 3, 2]), [-1, self.batch_size])
        final_state = tf.transpose(final_state, [1, 0])
        with tf.variable_scope('classifier'):
            W_final = tf.get_variable("W_final", [2 * num_layers * size, 4])
            b_final = tf.get_variable("b_final", [4])
        final_label = tf.matmul(final_state, W_final) + b_final
        return final_label

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def assign_anneal(self, session, anneal_value):
        session.run(self._anneal_update, feed_dict={self._new_anneal: anneal_value})

    @property
    def input(self):
        return self._input

    # @property
    # def initial_state(self):
    #     return self._initial_state
    #
    # @property
    # def encoded_state(self):
    #     return self._encoded_state
    #
    # @property
    # def cost(self):
    #     return self._recon_loss
    #
    # @property
    # def decoded_state(self):
    #     return self._decoded_state
    #
    # @property
    # def hidden_state(self):
    #     return self._hidden_state
    #
    # @property
    # def lr(self):
    #     return self._lr
    #
    # @property
    # def anneal(self):
    #     return self._anneal
    #
    # @property
    # def train_op(self):
    #     return self._train_op
    #
    # @property
    # def probabilities(self):
    #     return self._probabilities
    #
    # @property
    # def KL_loss(self):
    #     return self._KL_loss
    #
    # @property
    # def recon_loss(self):
    #     return self._recon_loss


