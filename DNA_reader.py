import tensorflow as tf
import numpy as np

# Token convention: GO 5, EOS 6, PAD 7


def DNA_producer(raw_data, batch_size, num_steps_encoder, num_steps_decoder, name=None):
    with tf.name_scope(name, "DNA_Producer", [raw_data, batch_size, num_steps_encoder, num_steps_decoder]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_size = tf.shape(raw_data)[0] #Total number of sequences
        num_batches = data_size // batch_size
        data = raw_data[0:batch_size*num_batches]

        GO_tensor = tf.constant(5, dtype=tf.int32, shape=[batch_size,1])
        EOS_tensor = tf.constant(6, dtype=tf.int32, shape=[batch_size,1])
        PAD_tensor = tf.constant(7, dtype=tf.int32, shape=[batch_size,num_steps_decoder-num_steps_encoder-2])
        PAD_tensor_target = tf.constant(7, dtype=tf.int32, shape=[batch_size,num_steps_decoder-num_steps_encoder-1])
        # data = tf.reshape(raw_data[0: batch_size * num_batches], [batch_size, num_batches])
        i = tf.train.range_input_producer(num_batches, shuffle= False).dequeue()
        encoder_input = tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder])
        decoder_input = tf.concat(1, [GO_tensor,
        						  tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder]),
        						  EOS_tensor,PAD_tensor])
        decoder_targets = tf.concat(1, [
        						  tf.slice(data,[i * batch_size, 0 ],[batch_size,num_steps_encoder]),
        						  EOS_tensor,PAD_tensor_target])

	return encoder_input, decoder_input, decoder_targets

def DNA_read(data_size=10000, seq_length=10, vocab_size=4):
	data_size_quarter = data_size // 4
	data_size = data_size_quarter * 4
	z1 = np.full([data_size_quarter,1],1, dtype=np.int64)
	for i in range(2,vocab_size+1):
	    y = np.full([data_size_quarter,1], i, dtype=np.int64)
	    z1 = np.concatenate((z1,y), axis=0)
	np.random.shuffle(z1)
	x = np.ones([data_size, seq_length]) * z1
	return x