import models
import utils
import tensorflow as tf

def main():
    with tf.Graph().as_default():
        with tf.Session() as session:
            model = models.DNA_seq_model()
            # model.train(session)
            # model.visualize_samples(session)
            model.load(session)
            model.report_performance(session)
            # model.visualize_samples_plot(session)
if __name__ == '__main__':
    main()