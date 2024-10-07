"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""
import os
import numpy as np
import scipy.misc
# from scipy.misc import imsave
# from keras.preprocessing.image import save_img as imsave
from skimage.io import imsave
from skimage.io import imread


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = int(rows), int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        h, w = int(h), int(w)
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        h, w = int(h), int(w)
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x
    
    imsave(save_path, img.astype('uint8'))


# write a simple test for this
def test_save_images():
    # Create a temporary directory for saving test images
    test_dir = 'test_images'
    os.makedirs(test_dir, exist_ok=True)

    # Test case 1: 2D array (grayscale images)
    X_2d = np.random.rand(16, 64*64)  # 16 images of size 64x64
    save_path_2d = os.path.join(test_dir, 'test_2d.png')
    save_images(X_2d, save_path_2d)
    assert os.path.exists(save_path_2d), "2D image file not created"

    # Test case 2: 3D array (grayscale images)
    X_3d = np.random.rand(16, 64, 64)  # 16 images of size 64x64
    save_path_3d = os.path.join(test_dir, 'test_3d.png')
    save_images(X_3d, save_path_3d)
    assert os.path.exists(save_path_3d), "3D image file not created"

    # Test case 3: 4D array (color images)
    X_4d = np.random.rand(16, 3, 64, 64)  # 16 color images of size 64x64
    save_path_4d = os.path.join(test_dir, 'test_4d.png')
    save_images(X_4d, save_path_4d)
    assert os.path.exists(save_path_4d), "4D image file not created"

    # Check if saved images have correct dimensions
    img_2d = imread(save_path_2d)
    img_3d = imread(save_path_3d)
    img_4d = imread(save_path_4d)

    assert img_2d.shape == (256, 256), "Incorrect shape for 2D case"
    assert img_3d.shape == (256, 256), "Incorrect shape for 3D case"
    assert img_4d.shape == (256, 256, 3), "Incorrect shape for 4D case"

    print("All tests passed successfully for tflib.save_images!")

    # Clean up: remove test directory and files
    for file in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, file))
    os.rmdir(test_dir)

# Run the test
if __name__ == "__main__":
    test_save_images()
