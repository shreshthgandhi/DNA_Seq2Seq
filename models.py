import tensorflow as tf
import numpy as np
import DNA_reader as reader
import time
import utils
import os

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
    data_size = 10000
    seq_length = 10


def get_config(flag='small'):
    if flag=='small':
        return SmallConfig()


def data_type():
    return tf.float32


class DNA_input(object):
    def __init__(self, config=None, data=None, name=None):
        if not(config):
            config = get_config()
        if not(data):
            data = reader.DNA_read(config.data_size, config.seq_length)

        self.batch_size = batch_size = config.batch_size
        self.num_steps_encoder = num_steps_encoder = config.num_steps_encoder
        self.num_steps_decoder = num_steps_decoder = config.num_steps_decoder
        self.epoch_size = len(data[0]) // batch_size
        (self.encoder_input, self.decoder_input,
         self.decoder_targets, self.labels) = reader.DNA_producer(data, config, name)


class DNA_seq_model(object):
    def __init__(self, config=None, input_=None):
        if not(config):
            config = get_config()
        if not(input_):
            input_ = DNA_input(config)
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
        # self.is_training = is_training
        self.config = config
        self.max_grad_norm = config.max_grad_norm
        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        #         weight_initializer = tf.random_normal_initializer(mean = 0.0, stddev=1.0)
        self.encoder_input = tf.placeholder(tf.int32, [None, self.num_steps_encoder])
        self.decoder_input = tf.placeholder(tf.int32,[None, self.num_steps_decoder])
        self.decoder_targets = tf.placeholder(tf.int32,[None, self.num_steps_decoder])
        self.labels = tf.placeholder(tf.int32, [None])

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.size, forget_bias=0.0, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
        self.initial_state = self.cell.zero_state(self.batch_size, data_type())
        temp_state = []
        with tf.variable_scope("Encoder_initial_state"):
            for i, (c, h) in enumerate(self.initial_state):
                temp = self.initial_state[i]._replace(c=tf.get_variable("init_c" + str(i), [self.batch_size, self.size]),
                                                       h=tf.get_variable("init_h" + str(i), [self.batch_size, self.size]))
                temp_state.append(temp)
        self.initial_state = tuple(temp_state)
        temp_state = []
        with tf.variable_scope("Decoder_initial_state"):
            for i, (c, h) in enumerate(self.initial_state):
                temp = self.initial_state[i]._replace(c=tf.get_variable("init_c" + str(i), [self.batch_size, self.size]),
                                                       h=tf.get_variable("init_h" + str(i), [self.batch_size, self.size]))
                temp_state.append(temp)
        self.decoder_initial_state = tuple(temp_state)
        #             self._initial_state[i].c = tf.get_variable("init_c"+str(i), [batch_size, size])
        #             self._initial_state[i].h = tf.get_variable("init_h"+str(i), [batch_size, size])

        #         lstm_cell_decoder = tf.nn.rnn_cell.BasicLSTMCell(size + compression_dims, forget_bias=0.0, state_is_tuple=True)
        #         decoder_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_decoder] * num_layers, state_is_tuple=True)
        #         decoder_initial_state = decoder_cell.zero_state(batch_size, data_type())
        with tf.variable_scope('Decoder'):
            softmax_w = tf.get_variable("softmax_w", [self.size, self.vocab_size], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=data_type())

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.vocab_size, self.size], dtype=data_type())
            self.encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
            self.decoder_inputs = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
        self.encoded_state = self.encoder(self.encoder_inputs)
        self.hidden_state, self.log_sigma_sq, self.mu = self.generate_hidden_state(self.encoded_state)
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

        self.final_state = tf.reshape(tf.transpose(tf.pack(self.decoder_state), [0, 1, 3, 2]), [-1, self.batch_size])
        self.final_state = tf.transpose(self.final_state, [1, 0])
        W_final = tf.get_variable("W_final", [2 * self.num_layers * self.size, 4])
        b_final = tf.get_variable("b_final", [4])
        self.final_label = tf.matmul(self.final_state, W_final) + b_final
        label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.final_label, self._input.labels)
        self._label_loss = tf.reduce_sum(label_loss) / self.batch_size

        decoder_output = tf.reshape(tf.concat(1, self.decoder_outputs), [-1, self.size])

        logits = tf.matmul(decoder_output, softmax_w) + softmax_b
        self.probabilities = tf.reshape(tf.nn.softmax(logits), [self.batch_size, self.num_steps_decoder, self.vocab_size])

        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._input.decoder_targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps_decoder], dtype=data_type())])
        #         KL_loss = 0.5 * (tf.reduce_sum(variance,1) + tf.reduce_sum(mu * mu,1) - compression_dims +tf.log(tf.reduce_prod(variance,1)) )
        KL_loss_batch = 0.5 * tf.reduce_sum(tf.exp(self.log_sigma_sq) + tf.square(self.mu) - self.log_sigma_sq - 1, 1)
        self.KL_loss =  tf.reduce_sum(KL_loss_batch) / self.batch_size

        self.recon_loss = Reconstruction_loss_avg = tf.reduce_sum(loss) / self.batch_size
        self.anneal = tf.Variable(0.0, trainable=False)
        self.cost = cost = self.recon_loss + self.anneal * self.KL_loss
        tf.scalar_summary("KL cost", self.KL_loss)
        tf.scalar_summary("Reconstruction cost", self.recon_loss)
        tf.scalar_summary("Total cost", self.cost)
        self.summary_op = tf.merge_all_summaries()

        # if not is_training:
        #     return

        self.lr = tf.Variable(0.0, trainable=False)
        #         self._anneal = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        #         optimizer = tf.train.AdamOptimizer(self._lr)
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self.global_step)

        self.new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.assign(self.lr, self.new_lr)
        self.new_anneal = tf.placeholder(
            tf.float32, shape=[], name='new_anneal')
        self.anneal_update = tf.assign(self.anneal, self.new_anneal)

    def train(self, session, config=None):
        # self.config = config
        if not(config):
            config = self.config
        if self.load(session, self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        self.writer = tf.train.SummaryWriter("./logs", session.graph)


        for i in range(config.max_max_epochs):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epochs, 0.0)
            anneal = (1 / (float(config.max_max_epochs - config.max_epochs))) * max(i + 1 - config.max_epochs, 0.0)
            self.assign_anneal(session, anneal)
            self.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f Anneal: %.3f" % (i + 1, session.run(self.lr), session.run(self.anneal)))
            train_perplexity = self.run_epoch(session, eval_op=self.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            self.save(session, self.checkpoint_dir, self.global_step)

    def visualize_samples(self, session):
        state = session.run(self.initial_state)
        self.teacher_force = False
        fetches = {
            "probabilities" : self.probabilities,
            "input" : self.input.encoder_input
        }
        inputs = []
        outputs = []
        for step in range(self.input.epoch_size):
            mbatch_encoder_input = self.input.encoder_input[self.batch_size * step:self.batch_size * (step + 1), :]
            mbatch_decoder_input = self.input.decoder_input[self.batch_size * step:self.batch_size * (step + 1), :]
            mbatch_decoder_targets = self.input.decoder_targets[self.batch_size * step:self.batch_size * (step + 1),
                                     :]
            mbatch_labels = self.input.labels[self.batch_size * step:self.batch_size * (step + 1)]
            feed_dict = {self.encoder_input: mbatch_encoder_input,
                         self.decoder_input: mbatch_decoder_input,
                         self.decoder_targets: mbatch_decoder_targets,
                         self.labels: mbatch_labels}
            vals = session.run(fetches, feed_dict)
            probs = vals["probabilities"]
            input_list = vals["input"].tolist()
            for seq in input_list:
                inputs.append(utils.num_to_string(seq))
            for i in range(self.batch_size):
                seq = []
                for j in range(self.num_steps_decoder):
                    # char = np.random.multinomial(1, probs[i, j]/(np.sum(probs[i,j])+1e-5))
                    char = np.argmax(char)
                    seq.append(char)
                outputs.append(utils.num_to_string(seq))
        for i in range(len(inputs)):
            print("%s\t\t%s" % (inputs[i], outputs[i]))

    def run_epoch(self, session, config=None, eval_op=None, verbose=False):
        if not config:
            config = self.config
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(self.initial_state)
        fetches = {"cost": self.cost,
                   "encoded_state":self.encoded_state,
                   "KL_loss":self.KL_loss,
                   "summary": self.summary_op}
        if eval_op is not None:
            fetches["eval_op"] = eval_op
        for step in range(self.input.epoch_size):
            mbatch_encoder_input = self.input.encoder_input[config.batch_size *step:config.batch_size*(step+1),:]
            mbatch_decoder_input = self.input.decoder_input[config.batch_size *step:config.batch_size*(step+1),:]
            mbatch_decoder_targets = self.input.decoder_targets[config.batch_size *step:config.batch_size*(step+1),:]
            mbatch_labels = self.input.labels[config.batch_size *step:config.batch_size*(step+1)]
            feed_dict={self.encoder_input:mbatch_encoder_input,
                       self.decoder_input:mbatch_decoder_input,
                       self.decoder_targets: mbatch_decoder_targets,
                       self.labels: mbatch_labels}
            vals = session.run(fetches, feed_dict)
            self.writer.add_summary(vals['summary'], self.global_step)
            costs += vals['cost']
            iters += self.num_steps_decoder
            if verbose and step % (self.input.epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f KL Loss: %.3f speed %.0f nps" %
                      (step * 1.0 / self.input.epoch_size, np.exp(costs / iters),
                       vals["KL_loss"],
                       iters * self.input.batch_size / (time.time() - start_time)))
            return np.exp(costs / iters)

    def save(self, session, checkpoint_dir, step):
        model_name = "DNAseq2seq.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(session, os.path.join(checkpoint_dir, model_name),
                        global_step = step)

    def load(self, session, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def encoder(self, encoder_inputs):
        # encoder_inputs = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
        # encoder_outputs = []
        state = self.initial_state
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
            W_decoder = tf.get_variable("W_decoder", [self.size + self.compression_dims, self.size], dtype=data_type())
            b_decoder = tf.get_variable("b_decoder", [self.size], dtype=data_type())

        for time_step in range(self.num_steps_decoder):
            decoder_inputs_list[time_step] = tf.matmul(decoder_inputs_list[time_step], W_decoder) + b_decoder

        def loop(prev, _):
            #             prev = tf.matmul(prev, softmax_w) + softmax_b
            #             prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            #             true_symbol =  tf.nn.embedding_lookup(embedding, prev_symbol)
            GO_token = tf.constant(vocab_size - 3, shape=[batch_size])
            true_symbol = tf.nn.embedding_lookup(self.embedding, GO_token)  # For GO token
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
        mu = tf.matmul(compressed_state, W_compress) +b_compress
        log_sigma_sq =  tf.matmul(compressed_state, W_sigma) + b_sigma
        hidden_state = mu + tf.sqrt(tf.exp(log_sigma_sq))* eps
        return hidden_state, mu, log_sigma_sq

    def classifier(self, decoder_state):
        final_state =  tf.reshape(tf.transpose(tf.pack(decoder_state), [0, 1, 3, 2]), [-1, self.batch_size])
        final_state = tf.transpose(final_state, [1, 0])
        with tf.variable_scope('classifier'):
            W_final = tf.get_variable("W_final", [2 * self.num_layers * self.size, 4])
            b_final = tf.get_variable("b_final", [4])
        final_label = tf.matmul(final_state, W_final) + b_final
        return final_label

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

    def assign_anneal(self, session, anneal_value):
        session.run(self.anneal_update, feed_dict={self.new_anneal: anneal_value})

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


