import numpy as np
import cv2
import os
import random

def compute_fourier_transform(image):
    F = np.fft.fft2(image)
    return F

def shift_ft(F):
    Fc = np.fft.fftshift(F)
    return Fc

def compute_magnitude(Fc):
    AF = np.abs(Fc)
    return AF

def get_max_freq(AF):
    M = np.max(AF)
    return M

def thresholding(AF,M):
    threshold = M/1000
    TH = np.sum(AF>threshold)
    return TH

def compute_quality_measure(image):
    '''Compute the Fourier Magnitude (FM) of an image.'''
    F = compute_fourier_transform(image)
    Fc = shift_ft(F)
    AF = compute_magnitude(Fc)
    M = get_max_freq(AF)
    TH = thresholding(AF,M)
    FM = TH/(image.shape[0]*image.shape[1])
    return FM

def convert_to_grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def apply_gauss_blur(image,kernel_size, sigma):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred

def load_random_image(folder_path):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image = random.choice(images)
    image_path = os.path.join(folder_path, image)
    image = cv2.imread(image_path)
    return image

def calc_FM(image):
    '''Calculate the Fourier Magnitude (FM) of an image with pre-processing'''
    gray_image = convert_to_grayscale(image)
    FM = compute_quality_measure(gray_image)
    return FM