"""
    Helper functions
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_images(images, cols=1, labels=None):
    """
       Plots a list of images

       Args:
        images (ndarray): Images to be plotted
        cols (int): Number of columns
        labels (list): List of labels with matching indexes to input images
    """
    n_images = len(images)
    if labels is None:
        labels = ["Image (%d)" % i for i in range(1, n_images + 1)]
    fig = plt.figure(figsize=(16, n_images))
    for n, (image, label) in enumerate(zip(images, labels)):
        axes = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        plt.axis("off")
        if image.ndim == 2 or image.shape[2] == 1:
            plt.gray()
        plt.imshow(image)
        axes.set_title(label)
    plt.show()
