"""
Monte Roybal
CS_575 Image Processing
Dr. Gil Gallegos
Homework 3 - Image Segmentation
"""

import cv2

"""Gaussian Blur Image Class"""
class Gauss_Blur():
    #Constructor Method
    def __init__(self,orig_image='',blur_image_obj=None,image_out=''):
        self.orig_image = cv2.imread(orig_image)
        self.blur_image_obj = blur_image_obj
        self.image_out = image_out
    #Print Method
    def str(self):
        return('Filename: {}'.format(self.orig_image))
    #Gaussian Blur Method
    def blur(self,img):
        self.blur_image = cv2.GaussianBlur(img,(7,7),0)
    #Remove icon Method
    def remove_icon(self):
        self.blur_image[200:360,1475:1640] = 128 
    #Write image Method
    def write_image(self,frame_num):
        self.image_out = cv2.imwrite('blurred_frame{}.png'.format(frame_num),self.blur_image) 

#Main Function
if __name__ == '__main__':
    for num in range(900,1900,100):
        blurred = Gauss_Blur('original_frame{}.png'.format(num))
        blurred.blur(blurred.orig_image)
        blurred.remove_icon()
        blurred.blur(blurred.blur_image)
        blurred.write_image(num)
        
        