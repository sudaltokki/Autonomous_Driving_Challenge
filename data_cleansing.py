import cv2
import os
import numpy

base_folder = "Compete_Dataset_Refined\\train\\OiSam03_Day"


for subfolder in os.listdir(base_folder):
    sub_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(sub_path):
        img_path = os.path.join(sub_path, 'img')
        new_txt_path = os.path.join(sub_path, 'new_txt')
        mask_path = os.path.join(sub_path, 'instance')
        cleansing_path = os.path.join(sub_path, 'cleansing')

        if not os.path.exists(cleansing_path):
            os.makedirs(cleansing_path)
            print('makedir ', cleansing_path)

        classes = ['c', 'b']
        locations = ['v', 'o', 'i', 'j', 'p']
        statuses = ['br', 'lft', 'rht', 'h']

        for image_file in os.listdir(img_path):
            if image_file.endswith('.jpg') or image_file.endswith('.png'):
                image_path = os.path.join(img_path, image_file)
                mask_image = os.path.join(mask_path, image_file)
                label_path = os.path.join(new_txt_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))

                image = cv2.imread(image_path)
                mask = cv2.imread(mask_image)
                h, w, _ = image.shape

                colorstep = numpy.array([60, 80, 110], dtype=numpy.uint8)
                mask *= colorstep

                with open(label_path, 'r') as file:
                    lines = file.readlines()

                for line in lines:
                    line = line.strip().split()
                    x1, y1, x2, y2 = map(int, line[:4])
                    class_id = int(line[4])
                    location_id = int(line[5])
                    statuses_flags = list(map(int, line[6:10]))

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"{classes[class_id]} | {locations[location_id]}"
                    for idx, status in enumerate(statuses_flags):
                        if status == 1:
                            label += f" | {statuses[idx]}"

                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                image = image//2 + mask//2

                output_path = os.path.join(cleansing_path, image_file)
                cv2.imwrite(output_path, image)

print("Done")
