import tensorflow as tf
import models

class LeNet:

    def __init__(self, lr, w_decay, keep_p, input_shape=[None, 28, 28], input_reshape=[-1, 28, 28, 1]):
        self.features = tf.placeholder(tf.float32, input_shape)
        self.images = tf.reshape(self.features, input_reshape)
        self.labels = tf.placeholder(tf.float32, [None, 10])
        self.labels = tf.cast(self.labels, tf.int32)
        self.learning_rate = lr
        
        with tf.variable_scope("LeNet") as scope:
            self.train_digits = models.lenet(images=self.images, dropout_keep_prob=keep_p, weight_decay=w_decay, is_training=True)
            scope.reuse_variables()
            self.pred_digits = models.lenet(images=self.images, dropout_keep_prob=keep_p, weight_decay=w_decay, is_training=False)
        self.logits = self.pred_digits
        self.correct_prediction = tf.equal(tf.argmax(self.pred_digits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.train_digits))
        self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss)