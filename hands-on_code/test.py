import argparse
from dataset import Dataset
import tensorflow as tf
import numpy as np
import learners
import generators
import coverage_analyzers as cov_anal

def test_model(checkpoint_path, model, n_elements, max_iterations, testing_dataset, coverage_analyzer):
    '''
    perform a metamorphic testing to TF-based model
    :param checkpoint_path: the path of checkpoint storing the model's parameters
    :param model: the model under test
    :param n_elements: number of elements sampled from the original test dataset
    :param max_iterations: max number of test cases generated from one original test input
    :param testing_dataset: the original test data
    :param coverage_analyzer: this component ensures measuring the coverage criteria 
    '''
    x_test, y_test = testing_dataset
    test_data = Dataset(x_test, y_test)
    data_shape = test_data.get_shape()
    sample_features, sample_labels = test_data.get_sample(sample_size=n_elements)
    prep_sample_features = test_data.standardize(sample_features)
    prep_sample_labels = test_data.get_one_hot_encoding(sample_labels)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        tensors = coverage_analyzer.get_tensors()
        outputs  = sess.run([model.accuracy, model.correct_prediction]+tensors, feed_dict={model.features: prep_sample_features, model.labels: prep_sample_labels})
        test_accurary = outputs[0]
        test_prediction = outputs[1]
        tensors_values = outputs[2:]
        coverage_analyzer.update_coverage(tensors_values)
        curr_coverage = coverage_analyzer.curr_coverage() 
        print('initial coverage: {}'.format(curr_coverage))
        print('test accurary: {}'.format(test_accurary))
        sample_features = sample_features[test_prediction]
        sample_labels = sample_labels[test_prediction]
        sample_size = np.sum(test_prediction)
        generator = generators.Generator(data_shape, sess, model, coverage_analyzer)
        for i in range(sample_size):
            print('running the example {}/{}'.format(i,sample_size))
            image = sample_features[i]
            curr_target = sample_labels[i]
            generator.run(image, curr_target, max_iterations)

def compute_empirical_bounds(model, training_dataset, coverage_analyzer):
    batch_size = 64
    x_train, y_train = training_dataset
    train_data = Dataset(x_train, y_train.squeeze(), reshape=False, one_hot=True, normalization=True)
    training_examples_count = train_data.get_num_examples()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(training_examples_count // batch_size):
            batch_x, batch_y = train_data.next_batch(batch_size)
            tensors = coverage_analyzer.get_tensors()
            tensors_values = sess.run(tensors, feed_dict={model.features: batch_x, model.labels: batch_y})
            coverage_analyzer.update_bounds(tensors_values)
            if i % 100 == 0:
                print('{} examples processed during empirical bounds determination'.format(i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help='help')
    parser.add_argument('--max', help='help')
    parser.add_argument('--cov', help='help')
    args = vars(parser.parse_args())
    n_elements = int(args['n'])
    max_iterations = int(args['max'])
    cov_criterion = args['cov']
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    model = learners.LeNet(lr=0.01,w_decay=0.0,keep_p=0.5)
    activations = ['LeNet/conv1/Relu:0', 'LeNet/conv2/Relu:0', 'LeNet/fc3/Relu:0']
    if cov_criterion=='nc':
        coverage_analyzer = cov_anal.NC(activations)
    elif cov_criterion=='kmnc':
        neurons_data_path = "./backup/lenet_neurons_data.bin"
        coverage_analyzer = cov_anal.KMNC(activations, data_path=neurons_data_path, k=100)
        if coverage_analyzer.neurons_data_exist():
            coverage_analyzer.load_neurons_data()
        else:
            compute_empirical_bounds(training_dataset=(x_train, y_train), model=model, coverage_analyzer=coverage_analyzer)
            coverage_analyzer.save_neurons_data()
    checkpoint_path = "./backup/lenet.ckpt"
    test_model(checkpoint_path=checkpoint_path, model=model, n_elements=n_elements, max_iterations=max_iterations, testing_dataset=(x_test, y_test), coverage_analyzer=coverage_analyzer)