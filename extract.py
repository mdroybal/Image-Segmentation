"""
Monte Roybal
CS_575 Image Processing
Dr. Gil Gallegos
Homework 3 - Image Segmentation
"""

import cv2

"""Extract Frames From Video Class"""
class Extract_Frames():
    #Constructor Method
    def __init__(self,step=100,count=0,success=True,video_obj="/home/monte/Desktop/CS_575/HW3_CS_575/Russia_Goes_Full_Kerbal.mp4"):
        self.step = step
        self.video_obj = cv2.VideoCapture(video_obj)
        self.count = count
        self.success = success
    #Print Method
    def __str__(self):
        return('Step: {}'.format(self.step))
    #Extract Frames(10) Method
    def extract(self):
        self.success,image = self.video_obj.read()
        if(self.success==True):
            if(self.count>=900 and self.count<=1800):
                if(self.count%self.step == 0):
                    cv2.imwrite('original_frame{}.png'.format(self.count),image)
            self.count+=1
        else:
            self.success = False

#Main Function
if __name__ == '__main__':
    frames = Extract_Frames()
    while frames.success:
        frames.extract()
        print(frames.success)
        print(frames.count)
    cv2.destroyAllWindows()
    frames.video_obj.release()