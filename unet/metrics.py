import numpy as np
import skimage

def logical_box_mask(box, img_dim=1024):
    """
    Input:
        box: [x, y, w, h] box coordinates
        img_dim: dimension of the image
    Output:
        (np.array of bool) mask
    """
    x, y, w, h = box
    mask = np.zeros((img_dim, img_dim), dtype=bool)
    mask[y:y+h, x:x+w] = True
    return mask

def parse_boxes(prediction_mask, threshold=0.2, min_box_area=0, connectivity=None):
    """
    Input:
        mask: (torch.Tensor) c x w x h tensor of the prediction mask
        threshold: pixel with value above threshold in the range 0-1 are considered positive target
        connectivity: None, 1 or 2 - connectivity parameter for skimage.measure.label segmentation
    Output:
        (list, list) of predicted_boxes, confidences
    """
    prediction_mask = prediction_mask[0]
    mask = np.zeros(prediction_mask.shape)
    mask[prediction_mask > threshold] = 1.
    
    label = skimage.measure.label(mask, connectivity=connectivity)
    
    predicted_boxes = []
    confidences = []
    for region in skimage.measure.regionprops(label):
        y1, x1, y2, x2 = region.bbox
        h = y2 - y1
        w = x2 - x1
        
        c = np.nanmean(prediction_mask[y1:y2, x1:x2])
        
        if w*h > min_box_area:
            predicted_boxes.append([x1, y1, w, h])
            confidences.append(c)
        
    return predicted_boxes, confidences

# plt.imshow(train_dataset[3][1][0], cmap=mpl.cm.jet) 
# print(train_dataset[3][1].shape)
# print(parse_boxes(train_dataset[3][1]))


def prediction_output(predicted_boxes, confidences):
    """
    Input:
        predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes' coordinates
        confidences: [c1, c2, ...] list of confidence values for the predicted boxes
    Output:
        'c1 x1 y1 w1 h1 x2 c2 y2 w2 h2 ...'
    """
    output = ''
    for c, box in zip(confidences, predicted_boxes):
        output += ' ' + str(c) + ' ' + ' '.join([str(b) for b in box])
    return output[1:]

def IoU(prediction_mask, ground_mask):
    """
    Input:
        prediction_mask: (numpy_array(bool)) predicted mask with bounding boxes masked as True
        ground_mask: (numpy_array(bool)) ground truth mask with bounding boxes masked as True
    Output:
        intersection(prediction_mask, ground_mask) / union(prediction_mask, ground_truth)
    """
    IoU = (prediction_mask & ground_mask).sum() / ((prediction_mask | ground_mask).sum() + 1.e-9)
    return IoU

def precision(tp, fp, fn):
    """
    Input:
        tp: (int) number of true positives
        fp: (int) number of false positives
        fn: (int) number of false negatives
    Output:
        precision
    """
    return float(tp) / (tp + fp + fn + 1.e-9)

def average_image_precision(predicted_boxes, confidences, target_boxes, img_dim=1024, min_box_area=0):
    """
    Input:
        predicted_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of predicted boxes' coordinates
        confidences: [c1, c2, ...] list of confidence values of the predicted boxes
        target_boxes: [[x1, y1, w1, h1], [x2, y2, w2, h2], ...] list of target boxes' coordinates
        img_dim: dimension of the boolean masks
        min_box_area: minimum area to be thresholded
    Output:
        Average precision
    """
    
    if len(predicted_boxes) == 0 and len(target_boxes) == 0:
        return np.nan
    else:
        if len(predicted_boxes) > 0 and len(target_boxes) == 0:
            return 0.0
        elif len(target_boxes) > 0 and len(predicted_boxes) == 0:
            return 0.0
        else:
            thresholds = np.arange(0.4, 0.8, 0.05)
            confidence_sorted_predicted_boxes = list(reversed(
                [b for _, b in sorted(zip(confidences, predicted_boxes),
                                      key=lambda pair:pair[0])]))
            average_precision = 0.0
            for thresh in thresholds:
                tp = 0
                fp = len(predicted_boxes)
                for predicted_box in confidence_sorted_predicted_boxes:
                    predicted_box_mask = logical_box_mask(predicted_box, img_dim)
                    for target_box in target_boxes:
                        target_box_mask = logical_box_mask(target_box, img_dim)
                        iou = IoU(predicted_box_mask, target_box_mask)
                        if iou > thresh:
                            tp += 1
                            fp -= 1
                            break

                fn = len(target_boxes)
                for target_box in target_boxes:
                    target_box_mask = logical_box_mask(target_box, img_dim)
                    for predicted_box in confidence_sorted_predicted_boxes:
                        predicted_box_mask = logical_box_mask(predicted_box, img_dim)
                        iou = IoU(predicted_box_mask, target_box_mask)
                        if iou > thresh:
                            fn -= 1
                            break
                
                average_precision += precision(tp, fp, fn) / float(len(thresholds))
            return average_precision
        
def average_batch_precision(output_batch, pids, boxes_by_pid_dict, rescale_factor, img_dim=1024, return_array=False, min_box_area=0):
    """
    Input:
        output_batch: output batch of the model
        pids: list of patient IDs in the batch
        boxes_by_pid_dict: dict of boxes given patient ID
        rescale_factor: rescale factor of the image
        img_dim: dimension of the image
        min_box_area: minimum area to be thresholded
    Output:
        average precision of the batch
    """
    
    batch_precisions = []
    for mask, pid in zip(output_batch, pids):
        target_boxes = boxes_by_pid_dict[pid] if pid in boxes_by_pid_dict else []
        if len(target_boxes) > 0:
            target_boxes = [[int(round(c/float(rescale_factor))) for c in target_box] for target_box in target_boxes]
        predicted_boxes, confidences = parse_boxes(mask, min_box_area=min_box_area)
        batch_precisions.append(average_image_precision(predicted_boxes, confidences, target_boxes, img_dim=img_dim))
        
    if return_array:
        return np.asarray(batch_precisions)
    else:
        return np.nanmean(np.asarray(batch_precisions))

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
        
    def __call__(self):
        return self.total / float(self.steps)
