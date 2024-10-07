import os
import numpy as np
import torch
import cv2
import ultralytics
import pandas
from pathlib import Path
#################################################################
# PARAMETERS
model = ultralytics.YOLO('./baseline_code/runs/nonE2E_Data/weights/best.pt').cuda()
image_directory = './data_new/images/train'
label_directory = './data_new/labels/train'
output_directory = './'
imgsz=[480,1280]
log_step=100
#################################################################
# [ car, bus, veh, out, inc, jun, prk, brk, lft, rgt, haz]
total_labels = np.zeros(11)
#################################################################
def update_total_classes(classes):
    if(classes[0] == 0):
        total_labels[0] += 1
    if(classes[0] == 1):
        total_labels[1] += 1
    if(classes[1] == 0):
        total_labels[2] += 1
    if(classes[1] == 1):
        total_labels[3] += 1
    if(classes[1] == 2):
        total_labels[4] += 1
    if(classes[1] == 3):
        total_labels[5] += 1
    if(classes[1] == 4):
        total_labels[6] += 1
    if(classes[2] == 1):
        total_labels[7] += 1
    if(classes[3] == 1):
        total_labels[8] += 1
    if(classes[4] == 1):
        total_labels[9] += 1
    if(classes[5] == 1):
        total_labels[10] += 1
#################################################################
def load_labels(label_folder, label_filename):
    labels = {'class': [], 'mask': []}
    with open(os.path.join(label_folder, label_filename)) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            classes = parts[:6]
            mask = parts[6:]
            update_total_classes(classes)
            labels['class'].append(classes)
            labels['mask'].append(mask)
    return labels
#################################################################
def load_images(image_folder, image_filename):
    img = cv2.imread(os.path.join(image_folder, image_filename))
    return img
#################################################################
def polygons_to_mask(polygons, size):
    masks = []
    for polygon in polygons:
        if(len(polygon) == 0):
            continue

        mask = np.zeros(size, dtype=np.uint8)
        contour = np.array(polygon).reshape((-1, 1, 2))
        
        contour *= [size[1], size[0]]
        contour = contour.astype(np.int32)
        
        cv2.fillPoly(mask, [contour], 1)
        masks.append(mask.astype(np.int32))
    
    nparray = np.array(masks)
    reshaped =  nparray.reshape(nparray.shape[0], -1) if nparray.size > 0 else np.array([])

    return torch.tensor(reshaped)
#################################################################
def compute_iou(gt_polygons, pred_polygons):
    iou_matrix = torch.zeros((len(gt_polygons), len(pred_polygons)))
    
    gt_masks = polygons_to_mask(gt_polygons, imgsz)
    pred_masks = polygons_to_mask(pred_polygons, imgsz)

    if(gt_masks.numel() > 0 and pred_masks.numel() > 0):
        iou_matrix = ultralytics.utils.metrics.mask_iou(gt_masks, pred_masks)  # Compute IoU

    return iou_matrix
#################################################################
def match_objects(gt_masks, pred_masks, iou_threshold=0.50):
    iou_matrix = compute_iou(gt_masks, pred_masks)
    
    matches = []
    
    gt_indices, pred_indices = torch.where(iou_matrix >= iou_threshold)

    matches = [(gt_index.item(), pred_index.item()) for gt_index, pred_index in zip(gt_indices, pred_indices)]

    matched_gt = set(gt_indices.tolist())
    matched_pred = set(pred_indices.tolist())

    full_gt = set(range(len(gt_masks)))
    full_pred = set(range(len(pred_masks)))

    missing_gt = full_gt - matched_gt
    missing_pred = full_pred - matched_pred

    return matches, missing_gt, missing_pred
#################################################################
def update_confusion_matrix(ground_truth, predicted, confusion_matrix, exclusive = True):
    if(exclusive):
        if(ground_truth == predicted):
            confusion_matrix[int(ground_truth)][0]+=1

        else:
            if(ground_truth >= 0):
                confusion_matrix[int(ground_truth)][1]+=1

            if(predicted >= 0):
                confusion_matrix[int(predicted)][2]+=1

    else:
        if(ground_truth == 0 and predicted == 0):
            return
        
        elif(ground_truth == predicted):
            confusion_matrix[0][0]+=1
        
        elif(ground_truth >= 0):
            confusion_matrix[0][1]+=1
        
        elif (predicted >= 0):
            confusion_matrix[0][2]+=1 
#################################################################
def confusion_to_array(confusion_matrix, total):
    TP, FN, FP = confusion_matrix.ravel()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    array = [TP, FN, FP, total, precision, recall]
    return array
#################################################################
def evaluate_model(image_folder, label_folder, output_folder):

    # TP, FN, FP
    cf_cls = np.zeros((2, 3))
    cf_loc = np.zeros((5, 3))
    cf_brk = np.zeros((1, 3))
    cf_lft = np.zeros((1, 3))
    cf_rgt = np.zeros((1, 3))
    cf_haz = np.zeros((1, 3))

    count = 0
    for image_filename in os.listdir(image_folder):
        if image_filename.endswith(('.jpg', '.png')):
            label_filename = image_filename.replace('.jpg', '.txt').replace('.png', '.txt')

            labels = load_labels(label_folder, label_filename)
            image = load_images(image_folder, image_filename)

            predictions = model(image,verbose=False, imgsz=[480,1280])

            predict_mask = []   

            if(bool(predictions[0].masks)):
                for premask in predictions[0].masks:
                    predict_mask.append(np.array(premask.xyn).flatten()) 
            
            matches, missing_gt, missing_pred = match_objects(labels['mask'], predict_mask)

            for match in matches:
                update_confusion_matrix(labels['class'][match[0]][0], predictions[0].boxes.data.cpu().numpy()[match[1],5].astype('int'), cf_cls)
                update_confusion_matrix(labels['class'][match[0]][1], predictions[0].boxes.data.cpu().numpy()[match[1],6].astype('int'), cf_loc)
                update_confusion_matrix(labels['class'][match[0]][2], predictions[0].boxes.data.cpu().numpy()[match[1],7].astype('int'), cf_brk, False)
                update_confusion_matrix(labels['class'][match[0]][3], predictions[0].boxes.data.cpu().numpy()[match[1],8].astype('int'), cf_lft, False)
                update_confusion_matrix(labels['class'][match[0]][4], predictions[0].boxes.data.cpu().numpy()[match[1],9].astype('int'), cf_rgt, False)
                update_confusion_matrix(labels['class'][match[0]][5], predictions[0].boxes.data.cpu().numpy()[match[1],10].astype('int'), cf_haz, False)
            
            for miss in missing_pred:
                update_confusion_matrix(-1, predictions[0].boxes.data.cpu().numpy()[miss,5].astype('int'), cf_cls)

            for miss in missing_gt:
                update_confusion_matrix(labels['class'][miss][0], -1, cf_cls)

            count+=1
            if(count % log_step == 0):
                print(str(count) + ' files done...')


    data = {
        'CAR': confusion_to_array(cf_cls[0], total_labels[0]),
        'BUS': confusion_to_array(cf_cls[1], total_labels[1]),
        'VEH': confusion_to_array(cf_loc[0], total_labels[2]),
        'OUT': confusion_to_array(cf_loc[1], total_labels[3]),
        'INC': confusion_to_array(cf_loc[2], total_labels[4]),
        'JUN': confusion_to_array(cf_loc[3], total_labels[5]),
        'PRK': confusion_to_array(cf_loc[4], total_labels[6]),
        'BRK': confusion_to_array(cf_brk[0], total_labels[7]),
        'LFT': confusion_to_array(cf_lft[0], total_labels[8]),
        'RGT': confusion_to_array(cf_rgt[0], total_labels[9]),
        'HAZ': confusion_to_array(cf_haz[0], total_labels[10]),
    }

    df = pandas.DataFrame(data, index=['TP', 'FN', 'FP', 'TOTAL', 'precision', 'recall'])
    print(df)

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(os.path.join(output_folder, 'evaluation_metrics.csv'))
#################################################################
if __name__ == '__main__':
    evaluate_model(image_directory, label_directory, output_directory)
#################################################################