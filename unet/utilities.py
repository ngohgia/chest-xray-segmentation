import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
import pylab
import shutil
from matplotlib.patches import Rectangle
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

DATA_DIR = '../data'

def get_patient_boxes(df, p_id):
    '''
    Input:
        df: data frame
        p_id: patient ID
    
    Output:
        array of all bouding boxes and the target's labels
        e.g:
        array([[x1, y1, width1, height1, class1, target1],
               [x2, y2, width2, height2, class2, target2]])
    '''
    boxes = df.loc[df['patientId'] == p_id][['x', 'y', 'width', 'height', 'class', 'Target']].values
    return boxes

def get_patient_boxes_values(df, p_id):
    '''
    Input:
        df: data frame
        p_id: patient ID
    
    Output:
        array of all bouding boxes and the target's labels
        e.g:
        array([[x1, y1, width1, height1],
               [x2, y2, width2, height2]])
    '''
    boxes = df.loc[df['patientId'] == p_id][['x', 'y', 'width', 'height']].values
    return boxes

def get_patient_dcm(p_id, sample='train'):
    '''
        p_id: patient ID
        sample: 'train' or 'test'
    '''
    return pydicom.read_file(os.path.join(DATA_DIR, 'stage_1_' + sample + '_images', p_id + '.dcm'))

def get_patient_metadata(p_id, attribute, sample='train'):
    '''
    Input:
        p_id: patient ID
        attribute: metadata's attribute
        sample: 'train' or 'test'
    Output:
        Value of metadata's attribute
    '''
    dcmdata = get_patient_dcm(p_id, sample=sample)
    attribute_value = getattr(dcmdata, attribute)
    return attribute_value

def display_patient_image(df, p_id, sample='train'):
    '''
    Input:
        df: dataframe
        p_id: patient ID
        sample: 'train' or 'test'
    Output:
        Display the patient's chest X-ray with overlaying bounding boxes and class annotation
    '''
    dcmdata = get_patient_dcm(p_id, sample=sample)
    dcmimg = dcmdata.pixel_array
    view = get_patient_metadata(p_id, 'ViewPosition', sample=sample)
    
    boxes = []
    if sample == 'train':
        boxes = get_patient_boxes(df, p_id)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(dcmimg, cmap=pylab.cm.binary)
    plt.axis('off')
    
    class_color_dict = { 'Normal': 'green',
                          'No Lung Opacity / Not Normal' : 'orange',
                          'Lung Opacity': 'red' }
    if len(boxes) > 0:
        for box in boxes:
            x, y, w, h, pclass, ptarget = box
            
            patch = Rectangle((x, y), w, h, color='red',
                            fill=False, lw=4, joinstyle='round', alpha=0.6)
            plt.gca().add_patch(patch)
            
    if sample == 'train':
        plt.text(10, 50, pclass, color=class_color_dict[pclass], size=20,
            bbox=dict(edgecolor=class_color_dict[pclass], facecolor='none', alpha=0.5, lw=2))
        plt.text(10, 100, view, color=class_color_dict[pclass], size=20,
                bbox=dict(edgecolor=class_color_dict[pclass], facecolor='none', alpha=0.5, lw=2))
    else:
        plt.text(10, 100, view, color='Red', size=20,
                bbox=dict(edgecolor='Red', facecolor='none', alpha=0.5, lw=2))

def min_max_scale_image(img, scale_range):
    '''
    Input:
        img: image
        scale_range: (tuple)(min, max) for scaling
    '''
    img = img.astype('float64')
    img_std = (img - np.min(img)) / (np.max(img) - np.min(img))
    scaled_img = img_std * float(scale_range[1] - scale_range[0]) + float(scale_range[0])
    scaled_img = np.rint(scaled_img).astype('uint8')

    return scaled_img

def elastic_transform_image(img, alpha, sigma, random_seed=None):
    '''
        Ref: https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a 
    '''
    assert len(img.shape) == 2, 'Image needs to be 2D'

    if random_seed is None:
        random_seed = np.random.RandomState(None)

    shape = img.shape
    
    dx = gaussian_filter((random_seed.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_seed.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    warped_image = map_coordinates(img, indices, order=1).reshape(shape)

    return warped_image

def rescale_box_coordinates(box, rescale_factor):
    x, y, w, h = box
    x = int(round(x/rescale_factor))
    y = int(round(y/rescale_factor))
    w = int(round(w/rescale_factor))
    h = int(round(h/rescale_factor))
    return [x, y, w, h]

def draw_boxes(predicted_boxes, confidences, target_boxes, ax, angle=0):
    if len(predicted_boxes)>0:
        for box, c in zip(predicted_boxes, confidences):
            x, y, w, h = box 

            patch = Rectangle((x,y), w, h, color='red',
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)

            ax.text(x+w/2., y-5, '{:.2}'.format(c), color='red', size=20, va='center', ha='center')
    if len(target_boxes)>0:
        for box in target_boxes:
            x, y, w, h = box
            patch = Rectangle((x,y), w, h, color='green',  
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            ax.add_patch(patch)
    
    return ax
