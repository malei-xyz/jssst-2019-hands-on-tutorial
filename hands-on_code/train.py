import tensorflow as tf
import os
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element
import argparse
from dataset import Dataset
import learners
import coverage_analyzers as cov_anal

def train_lenet(num_epoch, batch_size, learning_rate, weight_decay, keep_p, checkpoint_path):	
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_data = Dataset(x_train, y_train.squeeze(), reshape=False, one_hot=True, normalization=True)
    test_data = Dataset(x_test, y_test.squeeze(), reshape=False, one_hot=True, normalization=True)
    X_test, Y_test = test_data.get_data()
    num_sample = train_data.get_num_examples()
    model = learners.LeNet(lr=learning_rate, w_decay=weight_decay, keep_p=keep_p)
    saver = tf.train.Saver()
    best_test_accurary = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epoch):
            for i in range(num_sample // batch_size):
                batch_x, batch_y = train_data.next_batch(batch_size)
                sess.run(model.train_op, feed_dict={model.features: batch_x, model.labels: batch_y})
                if i % 10 == 0:
                    loss, accurary = sess.run([model.loss, model.accuracy],
                                        feed_dict={model.features: batch_x, model.labels: batch_y})
                    print('[Epoch {}] i: {} Loss: {} Accurary: {}'.format(epoch, i, loss, accurary))
            test_accurary = sess.run(model.accuracy, 
                                    feed_dict={model.features: X_test, model.labels: Y_test})
            print('Test Accurary: {}'.format(test_accurary))
            if best_test_accurary < test_accurary:
                saver.save(sess, checkpoint_path)
                best_test_accurary = test_accurary
            print('Best Test Accurary: {}'.format(best_test_accurary))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', help='help')
    parser.add_argument('--batch', help='help')
    parser.add_argument('--lr', help='help')
    parser.add_argument('--lambda', help='help')
    parser.add_argument('--keep', help='help')
    args = vars(parser.parse_args())
    num_epoch = int(args['epochs'])
    batch_size = int(args['batch'])
    learning_rate = float(args['lr'])
    weight_decay = float(args['lambda'])
    keep_p = float(args['keep'])
    if not os.path.isdir('./backup'):
        os.mkdir('./backup')
    checkpoint_path = "./backup/lenet.ckpt"
    train_lenet(num_epoch, batch_size, learning_rate, weight_decay, keep_p, checkpoint_path)
    