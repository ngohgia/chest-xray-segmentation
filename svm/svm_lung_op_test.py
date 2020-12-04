import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pylab
import pandas as pd
from glob import glob
import os.path as op
from sklearn import svm, metrics
import pickle
from skimage import measure
from scipy.stats import kurtosis, skew, entropy
import cv2 as cv
import csv

# Find bonding box

def bondingBox(label_p):
    b_box = []
    for i in range(1024):
        if (label_p[i] == -1 ):            
            b_box = np.append(b_box,[1,32*int(i%32),32*int(i/32),32,32])
    return b_box

# Patch you image
def patchScan_D(test_data,patch_size):
    
    featureNum = 9 # training data feature numbers
    patch = np.zeros((patch_size,patch_size))
    data_train_out = np.zeros((patch_size*patch_size,featureNum))
    
        
    for p in range(0,int(len(test_data)/patch_size)):
        for q in range(0,int(len(test_data)/patch_size)):
            patch = test_data[patch_size*p:patch_size*(p+1),patch_size*q:patch_size*(q+1)]
#                 print(i*1024+p*patch_size+q)
                                       
            # Feature extraction
            data_im = np.asarray(patch)
            data_row = patchFormat(patch)
            # No.1 Mean of Patch Image
            data_train_out[p*patch_size+q][0] = np.mean(data_im)
            # No.2 STD of Patch image
            data_train_out[p*patch_size+q][1] = np.std(data_im)
            # No.3 Shannon Entropy of Patch image
            def entropy1(labels, base=None):
                value,counts = np.unique(labels, return_counts=True)
                return entropy(counts, base=base)
            data_train_out[p*patch_size+q][2] = entropy1(data_im)
            # No.4 Skewness of Patch image
            if data_train_out[p*patch_size+q][1] == 0:
                data_train_out[p*patch_size+q][3] = 0
            else:
                data_train_out[p*patch_size+q][3] = skew(data_row,axis = 1)
            # No.5 Kurtosis of Patch image
            if data_train_out[p*patch_size+q][1] == 0:
                data_train_out[p*patch_size+q][4] = 0
            else:
                data_train_out[p*patch_size+q][4] = kurtosis(data_row,axis = 1)
            # Calculate X-axis direction and y-axis direction gradient
            sobelx = cv.Sobel(data_im,cv.CV_64F,1,0,ksize=5)
            sobely = cv.Sobel(data_im,cv.CV_64F,0,1,ksize=5)
            # No.6 Mean x-axis gradient of Patch image
            data_train_out[p*patch_size+q][5] = np.mean(sobelx)
            # No.7 STD x-axis gradient of Patch image
            data_train_out[p*patch_size+q][6] = np.std(sobelx)
            # No.8 Mean y-axis gradient of Patch image
            data_train_out[p*patch_size+q][7] = np.mean(sobely)
            # No.9 STD y-axis gradient of Patch image
            data_train_out[p*patch_size+q][8] = np.std(sobely)
         
                
    return data_train_out
    
# reshape patch from 32 by 32 to 1 by 1024    
def patchFormat(patch):
    patch = np.asarray(patch)
    pSize = len(patch)
    return np.reshape(patch,(1,pSize*pSize))    

def test_patient_data_Read(df):
 
    data = {}
    for n, row in df.iterrows():
        pid = row['patientId']
        if pid not in data:
            data[pid] = {
                'dicom': '%s/%s.dcm' % (images_test_dir, pid)
                }
    return data



def printBox(data,pids,model):

    ofile  = open('CSV_SVM_Lung_Opacity.csv', "w", newline='')   # create csv file
    writer = csv.writer(ofile, delimiter=',')    # begin writing to csv file
    writer.writerow(['patientId','PredictionString'])   # add header to csv file
    
    for pid in pids:
        
        d = pydicom.read_file(data[pid]['dicom'])
        im = d.pixel_array
        data_v_patch = patchScan_D(im,32)
        label_predict = model.predict(data_v_patch)
        print(label_predict)
        b_box = bondingBox(label_predict)
        entry = str(b_box)[1:-1].replace(',', '')   # format bounding box values
        writer.writerow([pid,entry])
        print(pid,b_box)
        
    ofile.close()   # stop writing to the csv file
    print('CSV OUTPUT Done!')
    return b_box



# Test Data Validation

images_test_dir = '../input/stage_1_test_images/'

images_test_df = pd.DataFrame({'path': glob(op.join(images_test_dir, '*.dcm'))})
images_test_df['patientId'] = images_test_df['path'].map(lambda x: op.splitext(op.basename(x))[0])
test_data_ids = list(images_test_df['patientId'])
data_validate = test_patient_data_Read(images_test_df)
print(len(data_validate))
print('Test Data Import Done!')
print(len(test_data_ids))


filename = 'Fature_SVM_model_test.sav'
clf = pickle.load(open(filename, 'rb'))
b_box = printBox(data_validate,test_data_ids[0:2],clf)