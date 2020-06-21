from PIL import Image
import numpy as np

def laplacianFilter(image):
    
    #Laplacian Filter
    laplacian = np.array([[0.3333, 0.3333, 0.3333], 
                          [0.3333, -2.6667, 0.3333], 
                          [0.3333, 0.3333, 0.3333]])
    
    sub_shape = (3,3)
    view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
    strides = image.strides + image.strides
    sub_matrices = np.lib.stride_tricks.as_strided(image,view_shape,strides)
    image = np.zeros((256,256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = np.einsum('ij,ij->', laplacian, sub_matrices[i][j])
        
    #image = image.astype('uint8')
    image = Image.fromarray(image)
    image.show()
    
    
def gaussianFilter(image):
    
    #Gaussian Filter
    gaussian = np.array([[0.0369, 0.0392, 0.0400, 0.0392, 0.0369], 
                         [0.0392, 0.0416, 0.0424, 0.0416, 0.0392], 
                         [0.0400, 0.0424, 0.0433, 0.0424, 0.0400],
                         [0.0392, 0.0416, 0.0424, 0.0416, 0.0392],
                         [0.0369, 0.0392, 0.0400, 0.0392, 0.0369]])
    
    sub_shape = (5,5)
    view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
    strides = image.strides + image.strides
    sub_matrices = np.lib.stride_tricks.as_strided(image,view_shape,strides)
    image = np.zeros((256,256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = np.einsum('ij,ij->', gaussian, sub_matrices[i][j])
        
        
    #image = image.astype('uint8')
    image = Image.fromarray(image)
    image.show()
    

def sobelFilter(image):
    
    #Sobel Filter
    sobel = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]])
    
    sub_shape = (3,3)
    view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
    strides = image.strides + image.strides
    sub_matrices = np.lib.stride_tricks.as_strided(image,view_shape,strides)
    image = np.zeros((256,256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = np.einsum('ij,ij->', sobel, sub_matrices[i][j])
        
        
    #image = image.astype('uint8')
    image = Image.fromarray(image)
    image.show()
    
    
def motionFilter(image):

    #Motion Filter
    motion = np.zeros((29, 29))
    motion[0][0] = 0.0156
    motion[0][1] = 0.0065
    motion[28][27] = 0.0065
    motion[28][28] = 0.0156
    for i in range(1,28):
        motion[i][i-1] = 0.0065
        motion[i][i] = 0.0223
        motion[i][i+1] = 0.0065
    
    sub_shape = (29,29)
    view_shape = tuple(np.subtract(image.shape, sub_shape) + 1) + sub_shape
    strides = image.strides + image.strides
    sub_matrices = np.lib.stride_tricks.as_strided(image,view_shape,strides)
    image = np.zeros((256,256))
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i][j] = np.einsum('ij,ij->', motion, sub_matrices[i][j])
        
        
    #image = image.astype('uint8')
    image = Image.fromarray(image)
    image.show()

def main():
    
    img = Image.open('cameraman.tif')

    
    #aimg_zero = np.pad(aimg, ((1,1), (1,1)), 'constant', constant_values=(0))
    
    #   LAPLACIAN FILTER WITH ZERO PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=1, mode='constant', constant_values=0)
    laplacianFilter(array_image)
    
    #   LAPLACIAN FILTER WITH WRAPPED PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=1, mode='wrap')
    laplacianFilter(array_image)
    
    #   GAUSSIAN FILTER WITH ZERO PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=2, mode='constant', constant_values=0)
    gaussianFilter(array_image)
    
    #   GAUSSIAN FILTER WITH WRAPPED PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=2, mode='wrap')
    gaussianFilter(array_image)
    
    #   SOBEL FILTER WITH ZERO PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=1, mode='constant', constant_values=0)
    sobelFilter(array_image)
    
    #   SOBEL FILTER WITH WRAPPED PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=1, mode='wrap')
    sobelFilter(array_image)
    
    #   MOTION FILTER WITH ZERO PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=14, mode='constant', constant_values=0)
    motionFilter(array_image)
    
    #   MOTION FILTER WITH WRAPPED PADDING
    array_image = np.array(img)
    array_image = np.pad(array_image, pad_width=14, mode='wrap')
    motionFilter(array_image)

main()