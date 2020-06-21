from PIL import Image, ImageDraw
import numpy as np
import math


def detect(testImage, templateImage):
    
    coords = [] 
    
    sub_shape = (templateImage.shape[0],templateImage.shape[1])
    view_shape = tuple(np.subtract(testImage.shape, sub_shape) + 1) + sub_shape
    strides = testImage.strides + testImage.strides
    sub_matrices = np.lib.stride_tricks.as_strided(testImage,view_shape,strides)
    
    
    for i in range(testImage.shape[0]-templateImage.shape[0]):
        for j in range(testImage.shape[1]-templateImage.shape[1]):
            
#            val = np.dot(templateImage, sub_matrices[i][j].T)/np.linalg.norm(templateImage)/np.linalg.norm(sub_matrices[i][j])
#            
#            val = np.sum(val)/val.shape[0]/val.shape[1]
#            
#            #print(val)
#            #check threshold
#            if(val >= threshold):
#                coords.append([i,j])
            
            val = calcDif(sub_matrices[i][j], templateImage)
            print(val)
            coords.append([i,j,val])
            
            
    index = np.where(coords == np.amin(coords))
    drawFace(coords[index][0], coords[index][1])
    
    

def calcDif(sub_image, templateImage):
    
    up=0
    d1=0
    d2=0
    
    meanTemp = np.sum(templateImage) / templateImage.shape[0] / templateImage.shape[1]
    meanSub = np.sum(sub_image) / sub_image.shape[0] / sub_image.shape[1]
    
    for s in range(templateImage.shape[0]):
        for r in range(templateImage.shape[1]):
            up += (templateImage[s][r] - meanTemp) * (sub_image[s][r] - meanSub)
            d1 += (templateImage[s][r] - meanTemp) ** 2
            d2 += (sub_image[s][r] - meanSub) ** 2
            
            #print(up, d1, d2)
    
    d1 = math.sqrt(d1)
    d2 = math.sqrt(d2)
    
    return up / (d1*d2)
    
def drawFace(x,y):
    
    
    #22 39
    image = Image.open('test_me.jpg').convert('RGBA')
    draw = ImageDraw.Draw(image)
                       
        
    temp = [x, y, x + 210, y + 256]
    draw.rectangle(temp, outline="#ff0000")   
    del draw
    image.show()
    image.save('output.bmp')
    
def main():
    
    #   test image gets ready to use         
    img = Image.open('test_me.jpg').convert('L')
    testImage = np.array(img)
    
    
    templateImage = np.zeros((256, 210))
    for i in range(1,4):
        img = Image.open("template_{}.jpg".format(i)).convert('L')
        temp = np.array(img)
        templateImage = np.add(templateImage, temp)
        
    #   template image gets ready to use 
    templateImage[:,:] = templateImage[:,:] / i
    img = Image.fromarray(templateImage)
    img.show()
    img.save('average_template.bmp')
    
    
    detect(testImage, templateImage)