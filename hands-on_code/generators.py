import os
import numpy as np
import random as rand
import matplotlib
import transformer as trans

class Generator(object):

    def __init__(self, shape, tf_session, tf_model, coverage_analyzer):
        self.session = tf_session
        self.model = tf_model
        self.tr_metadata = trans.build_transformation_metadata()
        self.adv_found = 0
        self.gen_examples = 0
        self.original_data = None
        self.target_class = None
        self.ssim_threshold = 0.75
        self.coverage_analyzer = coverage_analyzer
    
    # to complete
    def check_adv_objective(self, data, logits):
        '''
        check for fail tests and store misclassified examples
        :param data: a set of inputs
        :param logits: a set of logits (scores w.r.t labels for each input)
        '''
        pass

    def store_data(self, data, predict_class):
        self.adv_found += 1
        if not os.path.isdir('./adversarial_images'):
            os.mkdir('./adversarial_images')
        matplotlib.image.imsave("./adversarial_images/id_{}_class_{}.png".format(self.adv_found,predict_class), data)
    
    def standardize(self, features):
        axis = (0, 1, 2) if len(features.shape)==3 else (0, 1)
        mean_pixel = features.mean(axis=axis, keepdims=True)
        std_pixel = features.std(axis=axis, keepdims=True)
        return np.float32((features - mean_pixel) / std_pixel)
    
    def run(self, image, curr_target, max_iterations):
        '''
        generate synthetic inputs from one test input
        :param image: original test input
        :param curr_target: the ground truth label 
        :param max_iterations: max number of test cases generated from one original test input
        '''
        self.original_data = image
        self.target_class = curr_target
        for _ in range(max_iterations):
            transformed_data, ssim = trans.apply_random_transformation(self.original_data, self.tr_metadata)
            data = np.array(transformed_data)
            normalized_data = self.standardize(data)
            self.gen_examples += data.shape[0]
            if ssim > self.ssim_threshold:
                onehot_labels = np.zeros((data.shape[0],10))
                onehot_labels[:,self.target_class] = 1
                tensors = self.coverage_analyzer.get_tensors()
                outputs = self.session.run([self.model.logits]+tensors, feed_dict={self.model.features: normalized_data, self.model.labels: onehot_labels})
                logits = outputs[0]
                tensors_values = outputs[1:]
                self.coverage_analyzer.update_coverage(tensors_values)
                self.check_adv_objective(data, logits)
        print(self.coverage_analyzer.curr_coverage())
        print('{}/{}'.format(self.adv_found,self.gen_examples))
