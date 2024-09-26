import cv2
import os
import numpy

base_folder = "./data/train"

for subfolder in os.listdir(base_folder):
    sub_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(sub_path):
        img_path = os.path.join(sub_path, 'img')
        mask_path = os.path.join(sub_path, 'instance')
        cleansing_path = os.path.join(sub_path, 'masked')

        if not os.path.exists(cleansing_path):
            os.makedirs(cleansing_path)
            print('makedir ', cleansing_path)

        for image_file in os.listdir(img_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                base_image = cv2.imread(os.path.join(img_path, image_file))
                mask_image = cv2.imread(os.path.join(mask_path, image_file))

                colorstep = numpy.array([60, 80, 110], dtype=numpy.uint8) # to generate sudo-random colors
                mask_image *= colorstep
                base_image = base_image//2 + mask_image//2 
            
                output_path = os.path.join(cleansing_path, image_file)
                cv2.imwrite(output_path, base_image)
print("Done")
