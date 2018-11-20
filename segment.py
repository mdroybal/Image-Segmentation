"""
Monte Roybal
CS_575 Image Processing
Dr. Gil Gallegos
Homework 3 - Image Segmentation
"""

import cv2
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt

class Segment():
    #Constructor Method
    def __init__(self,orig_image='',num_colors=3):
        self.orig_image = cv2.imread(orig_image)
        self.num_colors = num_colors
        self.reshaped_image = None
        self.quantized_image = None
        self.labels = None
        self.rocket_bgr = []
        self.background_bgr = []
        self.foreground_bgr = []
        self.coordinates = []
        self.average_speed = []
        self.palette = 0
        self.width = 0
        self.height = 0
        self.depth = 0
    #Print Method
    def __str__(self):
        return('Number of Colors: {}'.format(self.num_colors))
    #Numpy reshape image Method
    def quantize(self):
        self.width,self.height,self.depth = self.orig_image.shape
        self.reshaped_image = np.reshape(self.orig_image,(self.width*self.height,self.depth))
    #Generate KMeans cluster model, labels, quantized image and palette with centroid's bgr values Method
    def cluster(self):
        model = cluster.KMeans(n_clusters=self.num_colors,random_state=4593)
        self.labels = model.fit_predict(self.reshaped_image)
        self.palette = model.cluster_centers_
        self.quantized_image = np.reshape(self.palette[self.labels],(self.width,self.height,self.palette.shape[1]))
    #Generate rocket image Method
    def gen_rocket_tensor(self,img):
        count = 0
        for y,x in self.coordinates:
            img[y,x] = 1
    #Plot quantized image Method
    def plot_image(self,img):
        plt.imshow(img)
        plt.draw()
        plt.show()
    #Plot x,y coordinates Method
    def plot_coordinates(self,coors):
        x_list = []
        y_list = []
        for y,x in coors:
            x_list.append(x)
            y_list.append(y)
        plt.plot(x_list,y_list,'ro',x_list,y_list,'k')
        plt.show()
    #Calculate average rocket speed Method
    def calc_average_speed(self,rock_coors):    
        x_diff = []
        y_diff = []
        for index,coors in enumerate(rock_coors):
            previous_x = rock_coors[index-1][1]
            previous_y = rock_coors[index-1][0]
            x_diff.append(abs(coors[1]-previous_x))
            y_diff.append(abs(coors[0]-previous_y))
        x_diff = x_diff[1:]
        y_diff = y_diff[1:]
        x_sum = sum(x_diff)
        y_sum = sum(y_diff)
        vx = float(x_sum)/(float(1800-900))
        vy = float(y_sum)/(float(1800-900))
        self.average_speed = [vx,vy]
        return(self.average_speed)
    #Calculate centroids with Numpy argwhere(to find x,y coordinates) Method
    def calc_centroids(self,bgr_vals):
        self.coordinates = np.argwhere(np.all(self.quantized_image==bgr_vals,axis=-1))
        y_centroid = 0
        x_centroid = 0
        for y,x in self.coordinates:
            y_centroid+=y
            x_centroid+=x
        return[((y_centroid/len(self.coordinates)),(x_centroid/len(self.coordinates)))]
    #Sort palette and assign bgr values to corresponding 'blobs' Method
    def sort_palette(self):
        for index,val in enumerate(self.palette):
            if(self.calc_bgr_avg(index) == self.bgr_avg_list()[0]):
                self.rocket_bgr.append(self.palette[index])
            if(self.calc_bgr_avg(index) == self.bgr_avg_list()[1]):
                self.background_bgr.append(self.palette[index])
            if(self.calc_bgr_avg(index) == self.bgr_avg_list()[2]):
                self.foreground_bgr.append(self.palette[index])
    #Calculate average bgr values across each 'blob' in palette Method
    def calc_bgr_avg(self,index):
        total = 0
        for val in self.palette[index]:
            total+=val
        return(total/len('bgr'))
    #Generate sorted list of average bgr values Method
    def bgr_avg_list(self):
        bgr_list = []
        for avg in range(3):
            bgr_list.append(self.calc_bgr_avg(avg))
        return(sorted(bgr_list))
    #Write image to file Method
    def write_image(self,img_file,img_obj):
        self.image_out = cv2.imwrite(img_file,img_obj)
    #Write data to file Method
    def write_file(self,fname,data):
        with open(fname,'w') as f:
            f.write(data)
        f.close 
#Main Function
if __name__ == '__main__':
    img_dims = Segment('original_frame900.png')
    img_dims.quantize()
    num_frames = 10
    nmd = (img_dims.width,img_dims.height,num_frames)
    rocket_image = np.zeros(nmd)
    rocket_coordinates = []
    background_coordinates = []
    foreground_coordinates = []
    rocket_tensor = []
    frame_count = 0
    f = open('rocket_tensor.txt','a')
    for num in range(900,1900,100):
        segmented = Segment('blurred_frame{}.png'.format(num))
        segmented.quantize()
        segmented.cluster()
        segmented.write_image('segmented_frame{}.png'.format(num),segmented.quantized_image)
        segmented.sort_palette()        
        rocket_coordinates+=segmented.calc_centroids(segmented.rocket_bgr)
        segmented.gen_rocket_tensor(rocket_image[:,:,frame_count])
        background_coordinates+=segmented.calc_centroids(segmented.background_bgr)
        foreground_coordinates+=segmented.calc_centroids(segmented.foreground_bgr)
        frame_count+=1
        if frame_count <= 9:
            np.savetxt(f,rocket_image[:,:,frame_count])
    f.close()
    print('Avergage speed {} pixels/frame'.format(segmented.calc_average_speed(rocket_coordinates)))
    segmented.plot_coordinates(rocket_coordinates)
    segmented.plot_coordinates(background_coordinates)
    segmented.plot_coordinates(foreground_coordinates)


