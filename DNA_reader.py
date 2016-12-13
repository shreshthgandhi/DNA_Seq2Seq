import tensorflow as tf
import numpy as np

# Token convention: GO 5, EOS 6, PAD 7


# def DNA_producer(raw_data, batch_size, num_steps_encoder, num_steps_decoder, name=None):
#     with tf.name_scope(name, "DNA_Producer", [raw_data, batch_size, num_steps_encoder, num_steps_decoder]):
#         raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
#         data_size = tf.shape(raw_data)[0]  # Total number of sequences
#         num_batches = data_size // batch_size
#         data = raw_data[0:batch_size * num_batches]

#         GO_tensor = tf.constant(5, dtype=tf.int32, shape=[batch_size, 1])
#         EOS_tensor = tf.constant(6, dtype=tf.int32, shape=[batch_size, 1])
#         PAD_tensor = tf.constant(7, dtype=tf.int32, shape=[batch_size, num_steps_decoder-num_steps_encoder - 2])
#         PAD_tensor_target = tf.constant(7, dtype=tf.int32, shape=[batch_size,num_steps_decoder-num_steps_encoder-1])
#         # data = tf.reshape(raw_data[0: batch_size * num_batches], [batch_size, num_batches])
#         i = tf.train.range_input_producer(num_batches, shuffle= False).dequeue()
#         encoder_input = tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder])
#         decoder_input = tf.concat(1, [GO_tensor,
#                                   tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder]),
#                                   EOS_tensor,PAD_tensor])
#         decoder_targets = tf.concat(1, [
#                                   tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder]),
#                                   EOS_tensor,PAD_tensor_target])

#     return encoder_input, decoder_input, decoder_targets


# def DNA_read(data_size=10000, seq_length=10, vocab_size=4):
#     data_size_quarter = data_size // 4
#     data_size = data_size_quarter * 4
#     z1 = np.full([data_size_quarter,1],1, dtype=np.int32)
#     for i in range(2,vocab_size+1):
#         y = np.full([data_size_quarter,1], i, dtype=np.int32)
#         z1 = np.concatenate((z1,y), axis=0)
#     np.random.shuffle(z1)
#     x = np.ones([data_size, seq_length]) * z1
#     return x

def DNA_read(data_size=10000, seq_length=10, vocab_size=4):
    data_size_quarter = data_size // 4
    data_size = data_size_quarter * 4
    x = []
    y = []
    for step in range(data_size):
        idx = step * vocab_size // (data_size)
        length = np.random.random_integers(seq_length // 2, seq_length)
        entry = np.ones([length], dtype=np.int64) * idx
        x.append(entry)
        y.append(idx)
    # np.random.shuffle(x)
    p = np.random.permutation(len(x))
    x = [x[i] for i in p]
    y = [y[i] for i in p]
    return x,y

def DNA_producer(raw_data, config, name=None):
    batch_size = config.batch_size
    num_steps_encoder = config.num_steps_encoder
    num_steps_decoder = config.num_steps_decoder
    vocab_size = config.vocab_size
    labels = raw_data[1]
    raw_data = raw_data[0]

    with tf.name_scope(name, "DNA_Producer"):
        data_size = len(raw_data)
        num_batches = data_size // batch_size
        data = raw_data[0: batch_size*num_batches]
        labels = labels[0: batch_size*num_batches]
        labels = tf.convert_to_tensor(labels, name="encoder_labels", dtype=tf.int32)
        # vocab_size -1 is 7 in our case and means PAD symbol
        encoder_input_full = np.ones([data_size, num_steps_encoder]) *(vocab_size-1)
        decoder_input_full = np.ones([data_size, num_steps_decoder]) *(vocab_size-1)
        decoder_targets_full = np.ones([data_size, num_steps_decoder]) *(vocab_size-1)
        for (i, seq) in enumerate(data):
            length = seq.shape[0]
            encoder_input_full[i, 0:length] = seq
            
            decoder_targets_full[i, 0:length] = seq
            decoder_targets_full[i, length] = vocab_size - 2  # For EOS
            
            decoder_input_full[i, 0] = vocab_size - 3
            decoder_input_full[i, 1:length + 1] = seq
            decoder_input_full[i, length + 1] = vocab_size - 2
        encoder_input_full = tf.convert_to_tensor(encoder_input_full, name="encoder_input", dtype=tf.int32)
        decoder_input_full = tf.convert_to_tensor(decoder_input_full, name="decoder_input", dtype=tf.int32)
        decoder_targets_full = tf.convert_to_tensor(decoder_targets_full, name="decoder_targets", dtype=tf.int32)
        i = tf.train.range_input_producer(num_batches, shuffle=False).dequeue()
        encoder_input = tf.slice(encoder_input_full, [i * batch_size, 0 ],[batch_size,num_steps_encoder])
        decoder_input = tf.slice(decoder_input_full, [i * batch_size, 0 ],[batch_size,num_steps_decoder])
        decoder_targets = tf.slice(decoder_targets_full, [i * batch_size, 0 ],[batch_size,num_steps_decoder])
        encoder_labels = tf.slice(labels, [i*batch_size],[batch_size])
    return encoder_input, decoder_input, decoder_targets, encoder_labels
