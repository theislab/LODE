import cv2
import numpy as np


class Segmentations():
    def __init__(self, dicom, model):
        self.dicom = dicom
        self.model = model
        self.oct_segmentations = self.get_oct_and_segmentation()

    def get_oct_and_segmentation(self):
        # read in oct
        self.oct_images = self.dicom.dicom_file.pixel_array

        # predict segmentation masks for all oct images
        oct_segmentations = []
        for i in range( 0, self.oct_images.shape[0] ):

            orig_height = 496
            orig_width = 512

            # stack oct image
            stacked_img = np.stack( (self.oct_images[i, :, :],) * 3, axis = -1 )

            # resize and scale stacked image
            resized_image = cv2.resize( stacked_img, (256, 256) ) / 255.

            # reshape image for prediction
            reshaped_image = resized_image.reshape( 1, 256, 256, 3 )
            prediction = cv2.resize(self.model.predict(reshaped_image )[0, :, :, 0],
                                                  (orig_height, orig_width),interpolation=cv2.INTER_NEAREST)

            # set class zero
            prediction[prediction< 0.5] = 0

            oct_segmentations.append(prediction)

        oct_segmentations = np.array( oct_segmentations )
        return oct_segmentations
