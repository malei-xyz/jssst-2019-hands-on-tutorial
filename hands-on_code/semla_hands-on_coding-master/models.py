import tensorflow as tf

slim = tf.contrib.slim

def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          weight_decay=0.0):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        activation_fn=tf.nn.relu):
        net = slim.conv2d(images, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        
        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout3')
        logits = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='fc4')
    return logits