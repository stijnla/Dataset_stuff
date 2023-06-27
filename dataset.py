import cv2
import numpy as np
import os
import copy


class ImageAugmenter():
    
    def __init__(self) -> None:
         pass
    

    @staticmethod     
    def _translate_left(image, percentage):
        filler = np.zeros((image.shape[0], int(image.shape[1]*percentage), image.shape[2]), dtype=np.uint8)
        translated_image = image[:, 0:int(image.shape[1]*(1-percentage)),:]
        translated_image = np.hstack((filler, translated_image))
        return translated_image


    @staticmethod  
    def _translate_right(image, percentage):
        percentage = -1*(1+percentage)
        filler = np.zeros((image.shape[0], int(image.shape[1]*(1+percentage)), image.shape[2]), dtype=np.uint8)
        translated_image = image[:, int(image.shape[1]*percentage):image.shape[1],:]
        translated_image = np.hstack((translated_image, filler))
        return translated_image


    @staticmethod  
    def _translate_up(image, percentage):
        filler = np.zeros((int(image.shape[0]*percentage), image.shape[1], image.shape[2]), dtype=np.uint8)
        translated_image = image[0:int(image.shape[0]*(1-percentage)),:,:]
        translated_image = np.vstack((filler, translated_image))
        return translated_image


    @staticmethod  
    def _translate_down(image, percentage):
        percentage = -1*(1+percentage)
        filler = np.zeros((int(image.shape[0]*(1+percentage)), image.shape[1], image.shape[2]), dtype=np.uint8)
        translated_image = image[int(image.shape[0]*percentage):image.shape[1],:,:]
        translated_image = np.vstack((translated_image, filler))
        return translated_image  
    
    
    @staticmethod
    def _rotate(image, degrees): 
        M = cv2.getRotationMatrix2D((image.shape[0]/2,image.shape[1]/2),degrees,1) 

        rotated_image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0])) 
        return rotated_image
    

class DatasetAugmenter(ImageAugmenter):

    def __init__(self, dataset) -> None:
        super().__init__()
        if type(dataset) == Dataset:
            self.original_dataset = dataset
        else:
            self.original_dataset = Dataset(dataset)

        self.augmented_dataset = Dataset(None)
    
    
    
    def rotate(self, image_path, annotation_path, degrees):
        """Rotates an image in speficied degrees in clockwise direction"""
        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path)

        rotated_image = super()._rotate(image, degrees)
        rotated_annotation = super()._rotate(annotation, degrees)
        return rotated_image, rotated_annotation



    def translateX(self, image_path, annotation_path, percentage):
        """Translates an image in specified percentage values in right direction"""
        if not (percentage <= 0.99 and percentage >= -0.99):
            raise(ValueError("Pick a value between -1 and 1"))

        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path)
        
        if percentage > 0:
            translated_image = super()._translate_left(image, percentage)
            translated_annotation = super()._translate_left(annotation, percentage)
        else:
            translated_image = super()._translate_right(image, percentage)
            translated_annotation = super()._translate_right(annotation, percentage)
        
        return translated_image, translated_annotation



    def translateY(self, image_path, annotation_path, percentage):
        """Translates an image in specified percentage values in upper direction"""
        if not (percentage <= 0.99 and percentage >= -0.99):
            raise(ValueError("Pick a value between -1 and 1"))
        percentage = -1*percentage
        image = cv2.imread(image_path)
        annotation = cv2.imread(annotation_path)
        
        if percentage > 0:
            translated_image = super()._translate_up(image, percentage)
            translated_annotation = super()._translate_up(annotation, percentage)
        else:
            translated_image = super()._translate_down(image, percentage)
            translated_annotation = super()._translate_down(annotation, percentage)
        
        return translated_image, translated_annotation
    


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