import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pylab
import pandas as pd
from glob import glob
import os.path as op
from sklearn import svm, metrics
from skimage import measure
from scipy.stats import kurtosis, skew, entropy
import cv2 as cv
import pickle


class_info_path = '../input/stage_1_detailed_class_info.csv'
train_labels_path = '../input/stage_1_train_labels.csv'
images_dir = '../input/stage_1_train_images/'

# data frames
class_info_df = pd.read_csv(class_info_path)
train_labels_df = pd.read_csv(train_labels_path)
images_df = pd.DataFrame({'path': glob(op.join(images_dir, '*.dcm'))})
images_df['patientId'] = images_df['path'].map(lambda x: op.splitext(op.basename(x))[0])
# parse DICOM header into dataframe
DICOM_TAGS = ['PatientAge', 'ViewPosition', 'PatientSex']
def get_tags(image_path):
    tag_data = pydicom.read_file(image_path, stop_before_pixels = True)
    tag_dict = {tag: getattr(tag_data, tag, '') for tag in DICOM_TAGS}
    tag_dict['path'] = image_path
    return pd.Series(tag_dict)
meta_df = images_df.apply(lambda x: get_tags(x['path']), 1)
meta_df['PatientAge'] = meta_df['PatientAge'].map(int)
meta_df.drop('path', 1).describe(exclude=np.number)

# concatenate the data frames
info_df = pd.concat([class_info_df, train_labels_df.drop('patientId', 1)], 1)
image_with_meta_df = pd.merge(images_df, meta_df, on='path')
bbox_with_info_df = pd.merge(info_df, image_with_meta_df, on='patientId', how='left')

bbox_with_info_df.sample(3)

def parse_patient_data(df):
    """
    Parse pandas dataframe into the following dictionary:
      data = {
        patientID: {
          'dicom': path/to/dicom/file,
          'target': 0 if normal, 1 if pneumonia,
          'boxes': list of box(es), each box is an array of number [x y width height],
          'class': one of the three values 'Lung Opacity', 'No Lung Opacity / Not Norma', 'Normal',
          'age': age of the patient,
          'view': either 'AP' - anteriorposterior, or 'PA' - posterioranterior,
          'sex': either 'Male' or 'Female'
        },
        ...
      }
    """
    
    extract_box = lambda row: [row['x'], row['y'], row['width'], row['height']]
    
    data = {}
    for n, row in df.iterrows():
        pid = row['patientId']
        if pid not in data:
            data[pid] = {
                'dicom': '%s/%s.dcm' % (images_dir, pid),
                'target': row['Target'],
                'class': row['class'],
                'age': row['PatientAge'],
                'view': row['ViewPosition'],
                'sex': row['PatientSex'],
                'boxes': []}
            
        if data[pid]['target'] == 1:
            data[pid]['boxes'].append(extract_box(row))
    return data

patients_data = parse_patient_data(bbox_with_info_df)
patient_ids = list(patients_data.keys())
#print(patients_data[np.random.choice(patient_ids)])
print('pydicom reading Done!')
for x in range(100,104):
    print(patients_data[patient_ids[x]])
print(len(patient_ids))
print(patient_ids[1])
z1 = patients_data[patient_ids[100]]['boxes']
print(len(z1))

# Get all lung opacity patients' ids 
data_op_ids = []
for npid in patient_ids:
    if patients_data[npid]['target'] == 1:
        data_op_ids = np.append(data_op_ids,npid)  


# reshape patch from 32 by 32 to 1 by 1024    
def patchFormat(patch):
    patch = np.asarray(patch)
    pSize = len(patch)
    return np.reshape(patch,(1,pSize*pSize))

# calculate the label of your patch
def patchLabel(patch_label,threshold):
    patch_label = np.asarray(patch_label)
    label_sum = np.sum(patch_label)
#    print(label_sum)
    label_out = 1
    if (label_sum <= threshold):
        label_out = -1
    
    return label_out

# Creare training List and corresponding labels
def data_assemble(patients_data, pids):
    
    data_train = [np.zeros((1024, 1024), dtype='uint8')] * len(pids)
    data_train_out = [np.zeros((1024, 1024), dtype='uint8')] * len(pids)
    label = [np.ones((1024, 1024),dtype='int')] * len(pids)
    i = 0;
    for pid in pids:
        d = pydicom.read_file(patients_data[pid]['dicom'])
        im = d.pixel_array
        data_train[i] = np.add(data_train[i], im)
        box_index = patients_data[pid]['boxes']
        if len(box_index) == 2:
            bx1 = np.asarray(box_index[0])
            bx1.astype(np.int64)
            bx2 = np.asarray(box_index[1])
            bx2.astype(np.int64)
            label[i][int(int(bx1[1])-1):int(bx1[1]+bx1[3]), int(int(bx1[0])-1):int(int(bx1[0] + bx1[2]))] = -1
            label[i][int(int(bx2[1])-1):int(bx2[1]+bx2[3]), int(int(bx2[0])-1):int(int(bx2[0] + bx2[2]))] = -1
        elif len(box_index) == 1:
            bx1 = np.asarray(box_index[0])
            bx1.astype(np.int64)
            label[i][int(bx1[1])-1:int(bx1[1])+int(bx1[3]), int(bx1[0])-1:int(bx1[2])+int(bx1[0])] = -1
        
        i = i + 1
        
    return data_train, label

# Segment image to 1024 pieces 32*32 patches and thresholding label sum for each pacth,
# Base on thresholding resuld, assign pacth label -1 for opacity or 1 for non-opacity.
def patchScan(test_data,test_label,patch_size,threshold):
    
    featureNum = 9 # training data feature numbers
    patch = np.zeros((patch_size,patch_size))
    data_train_out = np.zeros((len(test_data)*patch_size*patch_size,featureNum))
    label_train_out = np.zeros((len(test_data)*patch_size*patch_size,1))
    datatrain = np.zeros((1,patch_size*patch_size))
    patch_label = np.zeros((patch_size,patch_size))
    labeltrain = [[0]]
    
    for i in range(0,len(test_data)):        
        for p in range(0,int(len(test_data[i])/patch_size)):
            for q in range(0,int(len(test_data[i])/patch_size)):
                patch = test_data[i][patch_size*p:patch_size*(p+1),patch_size*q:patch_size*(q+1)]
#                 print(i*1024+p*patch_size+q)
                patch_label = test_label[i][patch_size*p:patch_size*(p+1),patch_size*q:patch_size*(q+1)]
                label_train_out[i*1024+p*patch_size+q] =  patchLabel(patch_label,threshold)
                
                # Feature extraction
                data_im = np.asarray(patch)
                data_row = patchFormat(patch)
                # No.1 Mean of Patch Image
                data_train_out[i*1024+p*patch_size+q][0] = np.mean(data_im)
                # No.2 STD of Patch image
                data_train_out[i*1024+p*patch_size+q][1] = np.std(data_im)
                # No.3 Shannon Entropy of Patch image
                def entropy1(labels, base=None):
                    value,counts = np.unique(labels, return_counts=True)
                    return entropy(counts, base=base)
                data_train_out[i*1024+p*patch_size+q][2] = entropy1(data_im)
                # No.4 Skewness of Patch image
                if data_train_out[i*1024+p*patch_size+q][1] == 0:
                    data_train_out[i*1024+p*patch_size+q][3] = 0
                else:
                    data_train_out[i*1024+p*patch_size+q][3] = skew(data_row,axis = 1)
                # No.5 Kurtosis of Patch image
                if data_train_out[i*1024+p*patch_size+q][1] == 0:
                    data_train_out[i*1024+p*patch_size+q][4] = 0
                else:
                    data_train_out[i*1024+p*patch_size+q][4] = kurtosis(data_row,axis = 1)
                # Calculate X-axis direction and y-axis direction gradient
                sobelx = cv.Sobel(data_im,cv.CV_64F,1,0,ksize=5)
                sobely = cv.Sobel(data_im,cv.CV_64F,0,1,ksize=5)
                # No.6 Mean x-axis gradient of Patch image
                data_train_out[i*1024+p*patch_size+q][5] = np.mean(sobelx)
                # No.7 STD x-axis gradient of Patch image
                data_train_out[i*1024+p*patch_size+q][6] = np.std(sobelx)
                # No.8 Mean y-axis gradient of Patch image
                data_train_out[i*1024+p*patch_size+q][7] = np.mean(sobely)
                # No.9 STD y-axis gradient of Patch image
                data_train_out[i*1024+p*patch_size+q][8] = np.std(sobely)
             
                
    return data_train_out, label_train_out

# Feature Extraction and Form New Training Dataset
def feature_extract(test_data,test_label,patch_size,threshold):
    data, label = data_assemble(test_data, test_label)
    d_train, label_train = patchScan(data,label,patch_size,threshold)
    label_train = np.ravel(label_train)
    return d_train, label_train

d_train, label_train = feature_extract(patients_data, patient_ids[0:20000],32,200)
# d_train, label_train = feature_extract(patients_data, data_op_ids[0:5000],32,200)
print('Feature Extraction Done!')

# SVM Classifier Trainning
def clf_data(test_data,test_label):

    clf_weights = svm.SVC(gamma=1)
    clf_weights.fit(test_data,test_label) #sample_weight = weight_matrix
    return clf_weights

clf_weights = clf_data(d_train,label_train)

# save classifier

filename = 'Fature_SVM_model_test.sav'
pickle.dump(clf_weights, open(filename, 'wb'))
print('SVM Trainning and Saving Done!')

# Test clasifier with training dataset
# d_v, l_v = feature_extract(patients_data, data_op_ids[5001:len(data_op_ids)],32,200)

d_v, l_v = feature_extract(patients_data, patient_ids[20000:22000],32,200)
l_predict = clf_weights.predict(d_v)
print("Classification report for classifier %s:\n%s\n"
      % (clf_weights, metrics.classification_report(l_v, l_predict)))
