import cv2
import numpy as np
import os
import copy
import scipy.stats as st


class ImageAugmenter():
    
    def __init__(self) -> None:
         pass
    
    @staticmethod
    def _adjust_brightness(image, value):
        """Adjust brightness of image, value between 0 and 1 decreases brightness, value between 1 and above increases brightness"""
       
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,2] = hsv[:,:,2]*value 
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image

    @staticmethod
    def _adjust_saturation(image, value):
        """Adjust saturation of image, value between 0 and 1 decreases saturation, value between 1 and above increases saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image

    @staticmethod
    def _channel_shift(image, value):
        """Shifts channels up or down depending on value, up when positive value, down when value is negative"""
        image = image +  (value, value, value)
        image[:,:,:][image[:,:,:]>255]  = 255
        image[:,:,:][image[:,:,:]<0]  = 0
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def _horizontal_flip(image):
        """Flips image horizontally"""
        return cv2.flip(image, 1)

    @staticmethod
    def _vertical_flip(image):
        """Flips image vertically"""
        return cv2.flip(image, 0)


    def _sharpen(self, image, value):
        """Sharpens image with specified value. Sharpening also depends on the blur kernel used"""
        # TODO: Bug with values greater than 255?
        blur = self._gaussian_kernel(255, 1)
        sharpened_image = np.float64(image) + value*(np.float64(image) - self._filter(np.float64(image), blur))
        sharpened_image = np.where(sharpened_image>255, 255, sharpened_image)
        sharpened_image = np.where(sharpened_image<0, 0, sharpened_image)

        return np.uint8(sharpened_image)


    def _gaussian_kernel(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel."""
        # increase nsig (variance) to make sharper)
        # increase kernlen (kernel size) to take more pixels into account
        x = np.linspace(-nsig, nsig, kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kern2d = np.outer(kern1d, kern1d)
        return kern2d/kern2d.sum()


    def _gaussianblur(self, image, value, variance=1):
        """Uses a gaussian kernel to blur the image"""
        gaussian_blur = self._gaussian_kernel(value, variance) 
        return self._filter(image, gaussian_blur)


    def _box_blur(self, image, value):
        """Uses a box kernel to blur the image"""
        box_blur= (1/(value**2))*np.ones((value, value))
        return self._filter(image, box_blur)


    def _emboss(self, image):
        """Creates edge enhancement in a certain direction"""
        #TODO: make variable
        emboss =  np.array([[-2, -1, 0],
                            [-1,  1, 1],
                            [ 0,  1, 2]])

        emboss2 =  np.array([[ 0,  1, 2],
                            [-1,  1, 1],
                            [-2, -1, 0]])
        
        emboss3 =  np.array([[2, 1, 0],
                            [1, 1, -1],
                            [0, -1, -2]])
        return self._filter(image, emboss)


    def _filter(self, image, filter):
        """Applies specified kernel to filter an image (blur, sharpen, etc.)"""
        filtered_image = cv2.filter2D(image,-1, filter)  
        return filtered_image


    def _add_colored_noise(self, image, stddev):
        """Adds colored noise to the image (random noise sampled for each channel)"""
        mean = 0
        noise = np.random.normal(mean, stddev,size=image.shape)
        noisy_image = np.float64(image) + np.float64(noise)
        noisy_image = np.where(noisy_image > 255, 255, noisy_image)
        noisy_image = np.where(noisy_image < 0, 0, noisy_image)
        return np.uint8(noisy_image)
    

    def _add_white_noise(self, image, stddev):
        """Adds white noise to the image (random noise sampled for all channels simaltaneously)"""
        mean = 0
        
        noise = np.random.normal(mean, stddev,size=(image.shape[0], image.shape[1], 1))
        noise = np.dstack((noise, noise, noise))
        noisy_image = np.float64(image) + np.float64(noise)
        noisy_image = np.where(noisy_image > 255, 255, noisy_image)
        noisy_image = np.where(noisy_image < 0, 0, noisy_image)
        return np.uint8(noisy_image)

    @staticmethod     
    def _translate_left(image, percentage):
        """Translates image left and fills with black background to maintain shape"""
        filler = np.zeros((image.shape[0], int(image.shape[1]*percentage), image.shape[2]), dtype=np.uint8)
        translated_image = image[:, 0:int(image.shape[1]*(1-percentage)),:]
        translated_image = np.hstack((filler, translated_image))
        return translated_image


    @staticmethod  
    def _translate_right(image, percentage):
        """Translates image right and fills with black background to maintain shape"""
        percentage = -1*(1+percentage)
        filler = np.zeros((image.shape[0], int(image.shape[1]*(1+percentage)), image.shape[2]), dtype=np.uint8)
        translated_image = image[:, int(image.shape[1]*percentage):image.shape[1],:]
        translated_image = np.hstack((translated_image, filler))
        return translated_image


    @staticmethod  
    def _translate_up(image, percentage):
        """Translates image up and fills with black background to maintain shape"""
        filler = np.zeros((int(image.shape[0]*percentage), image.shape[1], image.shape[2]), dtype=np.uint8)
        translated_image = image[0:int(image.shape[0]*(1-percentage)),:,:]
        translated_image = np.vstack((filler, translated_image))
        return translated_image


    @staticmethod  
    def _translate_down(image, percentage):
        """Translates image down and fills with black background to maintain shape"""
        percentage = -1*(1+percentage)
        filler = np.zeros((int(image.shape[0]*(1+percentage)), image.shape[1], image.shape[2]), dtype=np.uint8)
        translated_image = image[int(image.shape[0]*percentage):image.shape[1],:,:]
        translated_image = np.vstack((translated_image, filler))
        return translated_image  
    
    
    def _rotatedRectWithMaxArea(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
                return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)
        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = np.abs(np.sin(angle)), np.abs(np.cos(angle))

        if side_short <= 2.*sin_a*cos_a*side_long or np.abs(sin_a-cos_a) < 1e-10:
                # half constrained case: two crop corners touch the longer side,
                #   the other two corners are on the mid-line parallel to the longer line
                x = 0.5*side_short
                
                wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
                
        else:
                # fully constrained case: crop touches all 4 sides
                cos_2a = cos_a*cos_a - sin_a*sin_a
                wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr,hr

    
    def _rotate(self, image, angle): 
        """Rotates image, and fills background with black or returns the largest rectangle possible inside rotated image"""
        M = cv2.getRotationMatrix2D((image.shape[1]/2,image.shape[0]/2),angle,1) 

        rotated_image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0])) 
       
        maximum_width, maximum_height = self._rotatedRectWithMaxArea(rotated_image.shape[1], rotated_image.shape[0], np.pi*angle/180)
        height_padding = int((image.shape[0] - maximum_height)/2)
        width_padding = int((image.shape[1] - maximum_width)/2)
        if height_padding < 0:
                height_padding = 0
        rotated_image_no_fill = rotated_image[height_padding:rotated_image.shape[0]-height_padding, width_padding:rotated_image.shape[1]-width_padding]
        return rotated_image, rotated_image_no_fill



class DatasetAugmenter(ImageAugmenter):

    def __init__(self, dataset) -> None:
        super().__init__()
        if type(dataset) == Dataset:
            self.original_dataset = dataset
        else:
            self.original_dataset = Dataset(dataset)

        self.augmented_dataset = Dataset(None)
        self.augmentation_options = [self.random_brightness,
                                     self.random_saturation,
                                     self.random_channel_shift,
                                     self.random_sharpen_or_blur,
                                     self.random_add_noise,
                                     self.random_rotate,
                                     self.random_translateX,
                                     self.random_translateY,
                                     self.random_flip]
        

    def random_add_noise(self, image, annotation):
        colored = 1 if np.random.random() > 0.5 else 0
        stddev = 20 * np.random.random()
        if colored:
            return super()._add_colored_noise(image, stddev), annotation
        else:
            return super()._add_white_noise(image, stddev), annotation

    def random_channel_shift(self, image, annotation):
        strength = int(255 * np.random.random() - 128)
        return super()._channel_shift(image, strength), annotation

    def random_brightness(self, image, annotation):
        strength = 2 * np.random.random()
        return super()._adjust_brightness(image, strength), annotation    

    def random_saturation(self, image, annotation):
        strength = 2 * np.random.random()
        return super()._adjust_saturation(image, strength), annotation    

    def random_sharpen_or_blur(self, image, annotation):
        sharpen = 1 if np.random.random() > 0.5 else 0
        if sharpen:
            strength = np.random.random()
            return super()._sharpen(image, strength), annotation
        else:
            gaussian = 1 if np.random.random() > 0.5 else 0
            kernel_size = 2*np.random.randint(0, 50) + 1
            return self.add_blur(image, kernel_size, gaussian), annotation
        
    def random_rotate(self, image, annotation):
        angle = np.random.randint(0, 360)
        return self.rotate(image, annotation, angle)

    def random_translateX(self, image, annotation):
        percentage = 0.5*np.random.random() - 0.5
        return self.translateX(image, annotation, percentage)

    def random_translateY(self, image, annotation):
        percentage = 0.5*np.random.random() - 0.5
        return self.translateY(image, annotation, percentage)
    
    def random_flip(self, image, annotation):
        if np.random.randint(1):
            direction = 'vertical'
        else:
            direction = 'horizontal'
        return self.flip(image, annotation, direction)
    

    def add_blur(self, image, kernel_size, gaussian):
        if gaussian:
            variance = 1
            image = super()._gaussianblur(image, kernel_size, variance)
        else:
            image = super()._box_blur(image, kernel_size)
        return image
    

    def add_noise(self, image, stddev, colored):
        
        if colored:
            image = super()._add_colored_noise(image, stddev)
        else:
            image = super()._add_white_noise(image, stddev)
        return image


    def translateX(self, image, annotation, percentage):
        """Translates an image with annotation in specified percentage values in right direction"""
        if not (percentage <= 0.99 and percentage >= -0.99):
            raise(ValueError("Pick a value between -1 and 1"))
        
        if percentage > 0:
            translated_image = super()._translate_left(image, percentage)
            translated_annotation = super()._translate_left(annotation, percentage)
        else:
            translated_image = super()._translate_right(image, percentage)
            translated_annotation = super()._translate_right(annotation, percentage)
        
        return translated_image, translated_annotation


    def translateY(self, image, annotation, percentage):
        """Translates an image with annotation in specified percentage values in upper direction"""
        if not (percentage <= 0.99 and percentage >= -0.99):
            raise(ValueError("Pick a value between -1 and 1"))
        percentage = -1*percentage
        
        if percentage > 0:
            translated_image = super()._translate_up(image, percentage)
            translated_annotation = super()._translate_up(annotation, percentage)
        else:
            translated_image = super()._translate_down(image, percentage)
            translated_annotation = super()._translate_down(annotation, percentage)
        
        return translated_image, translated_annotation
    

    def rotate(self, image, annotation, angle):
        """Rotates an image with annotation in speficied degrees in clockwise direction"""

        rotated_image, rotated_no_fill_image = super()._rotate(image, angle)
        rotated_annotation, rotated_no_fill_annotation = super()._rotate(annotation, angle)

        nofill = True
        if nofill:
            return rotated_no_fill_image, rotated_no_fill_annotation
        else:
            return rotated_image, rotated_annotation


    def flip(self, image, annotation, direction):
        """Flips an image with annotation in speficied direction"""

        if direction == 'horizontal':
            flipped_image = super()._horizontal_flip(image)
            flipped_annotation = super()._horizontal_flip(annotation)
        elif direction == 'vertical':
            flipped_image = super()._vertical_flip(image)
            flipped_annotation = super()._vertical_flip(annotation)
        else:
            raise ValueError("Direction must be either vertical or horizontal, not '" + str(direction) + "'")
        
        return flipped_image, flipped_annotation
    

    def test(self):
        for i, train_image in enumerate(self.original_dataset.data['train']['images']):
            train_annotation = self.original_dataset.data['train']['annotations'][i]
            image = cv2.imread(train_image)
            annotation = cv2.imread(train_annotation)
            self.augment(image, annotation)

    def augment(self, image, annotation):
        [f(image, annotation) for f in self.augmentation_options]


class Dataset():

    def __init__(self, path_to_dataset):
        """Load dataset into dictionary called data"""
        
        self.modes = ['train', 'validation']

        self.data = {}

        if path_to_dataset:
            self.load_dataset(path_to_dataset)
        else:
            # Generate empty dataset
            mode_data = {'images':[], 'annotations':[]}
            for mode in self.modes:
                self.data[mode] = mode_data



    def load_dataset(self, path_to_dataset):
        success = True

        for mode in self.modes:
            mode_data = {}

            try:
                data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', path_to_dataset, mode)
                image_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file))]
            except:
                success = False
                image_files = []

            mode_data['images'] = image_files

            try:
                annotation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', path_to_dataset, mode + "_annotation")
                annotation_files = [os.path.join(annotation_path, file) for file in os.listdir(annotation_path) if os.path.isfile(os.path.join(annotation_path, file))]
            except:
                success = False
                annotation_files = []

            mode_data['annotations'] = annotation_files
            
            self.data[mode] = mode_data
        
        if success:
            print("Succesfully loaded dataset")
        else:
            print("WARNING: could not find dataset, loaded empty dataset")
        


    def convert_rgb_segmentation_to_binary_mask(self, segmentation):
        binary_annotation =  cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(binary_annotation, 128, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask



    def get_segmentation_mask(self, annotation, color):
        mask = self.convert_rgb_segmentation_to_binary_mask(annotation)
        annotation = copy.deepcopy(mask)
        # convert each to a color
        annotation[np.where((annotation==[255, 255, 255]).all(axis=2))] = color
        return cv2.cvtColor(annotation, cv2.COLOR_BGR2BGRA), mask
 


    def view_annotated_image(self, image_file, annotation_file):
        
        red = [0, 0, 255] # visible color underwater
        
        # read files, convert annotation to a mask and colored mask
        annotation = cv2.imread(annotation_file)
        image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2BGRA)
        colored_mask, binary_mask = self.get_segmentation_mask(annotation, red)
        
        number_of_pipe_pixels = np.sum(binary_mask == 255)
    
        # visualize colored pipe in image
        combined_image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)
        cv2.imshow('image', combined_image)
        cv2.waitKey(10)          



    def view_data(self, mode):
        assert mode in self.modes

        for i, image_file in enumerate(self.data[mode]['images']):
            annotation_file = self.data[mode]['annotations'][i]
            
            self.view_annotated_image(image_file, annotation_file)

d = Dataset("new_dataset")
print(len(d.data['train']['images']) + len(d.data['validation']['images']))
#d.view_data('validation')
b = DatasetAugmenter(d)
b.test()