import numpy as np
from PIL import ImageEnhance
from PIL import Image
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import transform
from skimage import io
import tensorflow as tf
import matplotlib
import os
from skimage.measure import compare_ssim as ssim
import argparse

def rotate(img, rad_angle):
    afine_tf = transform.AffineTransform(rotation=rad_angle)
    rotated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return rotated_img

def translate(img, trans_x, trans_y):
    afine_tf = transform.AffineTransform(translation=(trans_x, trans_y))
    translated_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return translated_img

def scale(img, scale_1, scale_2):
    afine_tf = transform.AffineTransform(scale=(scale_1, scale_2))
    scaled_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return scaled_img

def shear(img, value):
    afine_tf = transform.AffineTransform(shear=value)
    sheared_img = np.uint8(transform.warp(img, inverse_map=afine_tf, preserve_range=True))
    return sheared_img
    
def blur(img, sigma):
    is_colour = len(img.shape)==3
    blur_img = np.uint8(rescale_intensity(gaussian(img, sigma=sigma, multichannel=is_colour,preserve_range=True),out_range=(0,255)))
    return blur_img

def change_brightness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Brightness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_color(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Color(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_contrast(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Contrast(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def change_sharpness(img, factor):
    image = Image.fromarray(img)
    data = np.array(ImageEnhance.Sharpness(image).enhance(factor).getdata(), dtype="uint8").reshape(img.shape)
    return data

def apply_mutation(img, delta, mutation_prob):
    '''
    apply pixel perturbation to an image
    :param delta: the pixel changes could be a random value sampled from [-delta,delta]
    :param mutation_prob: the probability of applying a perturbation to one pixel
    :return: a mutated image
    '''
    normalized_img = normalize(img)
    shape = img.shape
    U = np.random.uniform(size=shape)*2*delta - delta
    mask = np.random.binomial(1, mutation_prob, size=shape)
    mutation = mask * U
    mutated_img = normalized_img + mutation
    denormlized_img = denormlize(mutated_img)
    return denormlized_img

def build_transformation_metadata(): 
    '''
    construct a dictionary containing the interval boundaries for valid transformations' parameters
    :return: the dictionary of transformations' metadata
    '''
    tr_metadata = {
        'sigma_min_bound':0.0,
        'sigma_max_bound':0.5,
        'contrast_min_bound':1.0,
        'contrast_max_bound':2.0,
        'brightness_min_bound':1.0,
        'brightness_max_bound':2.0,
        'sharpness_min_bound':1.0,
        'sharpness_max_bound':2.0,
        'delta':0.05,
        'mutation_prob': 0.01,
        'scale_max_bound': 1.0,
        'scale_min_bound': 0.95,
        'trans_abs_bound': 4, # absolute bound means ==> min: -4, max: 4
        'shear_abs_bound':0.1,
        'rot_angle_abs_bound': np.pi/45
    }
    return tr_metadata

# to complete
def apply_random_transformation(image_origin, tr_metadata):
    '''
    apply a sequence of image transformations to the image 
    :return transformed_data: a set of transformed images
    :return ssim_value: the ssim value comparing between the original image and the transformed one w.r.t pixel-value transformations
    '''
    transformed_data = []
    # apply mutation using the metadata parameters 
    
    # change contrast using random factor within the valid interval

    # change brightness using random factor within the valid interval

    # change sharpness using random factor within the valid interval

    # add blur effect using random sigma within the valid interval

    # compute the ssim between the original image and the resulting pixel-value transformed image

    # add the transformed image to the transformed data

    # translate the image using random translation parameters within the valid interval

    # add the translated image to the transformed data

    # scale the image using random scale parameters within the valid interval

    # add the scaled image to the transformed data

    # rotate the image using random angle  within the valid interval

    # add the rotated image to the transformed data

    # shear the image using random value within the valid interval

    # add the sheared image to the transformed data

def normalize(img):
    norm_img = np.float32(img / 255.0)
    return norm_img

def denormlize(img):
    denorm_img = np.uint8(img * 255.0)
    return denorm_img

def store_data(id, data):
    if not os.path.isdir('./test_images'):
        os.mkdir('./test_images')
    matplotlib.image.imsave("./test_images/id_{}.png".format(id), data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', help='help')
    parser.add_argument('--attempts', help='help')
    args = vars(parser.parse_args())
    ssim_threshold = float(args['threshold'])
    attempts_count = int(args['attempts'])
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    tr_meta = build_transformation_metadata()
    for i in range(attempts_count):
        image_idx = np.random.choice(len(x_test))
        image = x_test[image_idx]
        mutated_images, ssim_value = apply_random_transformation(image, tr_meta)
        if ssim_value > ssim_threshold:
            store_data(i, image)
      