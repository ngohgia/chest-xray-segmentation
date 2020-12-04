import os
import time
import pickle

import torch
import numpy as np

from utilities import *
from metrics import *
from pneumonia_dataset import load_data
from unet import LeakyUNET
from loss import BCEWithLogitLoss2D
from experiment import *

# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'

EXP_NAME = 'UNET_SingleView_Baseline'
timestamp = time.strftime('%m%d-%H%M%S', time.localtime())
output_dir = os.path.join('./output/', EXP_NAME, timestamp)
os.makedirs(output_dir)

debug = False
original_dim = 1024

# TODO: change rescale_factor or batch_size
rescale_factor = 4
batch_size = 32
validation_prop = 0.1

data_dir = './data'
train_csv_path = os.path.join(data_dir, 'train_validation_test', 'train.csv')
test_csv_path = os.path.join(data_dir, 'train_validation_test', 'test.csv')

train_images_dir = os.path.join(data_dir, 'stage_1', 'stage_1_train_images')
test_images_dir = os.path.join(data_dir, 'stage_1', 'stage_1_test_images')

train_df, train_loader, dev_pids, dev_loader, dev_dataset_for_predict, dev_loader_for_predict, test_loader, test_df, test_pids, boxes_by_pid_dict, min_box_area = load_data(train_csv_path, test_csv_path, train_images_dir, test_images_dir, batch_size, validation_prop, rescale_factor)
min_box_area = int(round(min_box_area / float(rescale_factor**2)))

# model = torch.nn.DataParallel(LeakyUNET().cuda(), device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
model = torch.nn.DataParallel(LeakyUNET().cuda(), device_ids=[0, 1, 2, 3])

loss_fn = BCEWithLogitLoss2D().cuda()

init_learning_rate = 0.5

num_epochs = 2 if debug else 10
num_train_steps = 5 if debug else len(train_loader)
num_dev_steps = 5 if debug else len(dev_loader)

img_dim = int(round(original_dim / rescale_factor))

print("Training for {} epochs".format(num_epochs))
histories, best_models = train_and_evaluate(model, train_loader, dev_loader, init_learning_rate,
                                          loss_fn, num_epochs, num_train_steps, num_dev_steps,
                                          boxes_by_pid_dict, rescale_factor, img_dim, output_dir, min_box_area=min_box_area)

# print('- Predicting with best PRECISION model')
# best_precision_model = best_models['best precision model']
# torch.save(best_precision_model, os.path.join(output_dir, 'best_precision_model.pt'))
# dev_predictions = predict(best_precision_model, dev_loader_for_predict)
# test_predictions = predict(best_precision_model, test_loader)
# 
# box_thresh = best_box_thresh_from_dev_predictions(dev_predictions, dev_dataset_for_predict, rescale_factor, min_box_area, boxes_by_pid_dict)
# # box_thresh = 0.2
# print('Best thresh ' + str(box_thresh))
# save_predictions_to_csv(test_df, test_pids, test_predictions, box_thresh, min_box_area, rescale_factor, output_dir, 'best_precision')
# save_predictions_to_csv(train_df, dev_pids, dev_predictions, box_thresh, min_box_area, rescale_factor, output_dir, 'DEV_best_precision')
# 
print('- Predicting with best LOSS model')
best_loss_model = best_models['best loss model']
torch.save(best_loss_model, os.path.join(output_dir, 'best_loss_model.pt'))
dev_predictions = predict(best_loss_model, dev_loader_for_predict)
test_predictions = predict(best_loss_model, test_loader)

box_thresh = best_box_thresh_from_dev_predictions(dev_predictions, dev_dataset_for_predict, rescale_factor, min_box_area, boxes_by_pid_dict)
print('Best thresh ' + str(box_thresh))
# box_thresh = 0.2
save_predictions_to_csv(test_df, test_pids, test_predictions, box_thresh, min_box_area, rescale_factor, output_dir, 'best_loss')
save_predictions_to_csv(train_df, dev_pids, dev_predictions, box_thresh, min_box_area, rescale_factor, output_dir, 'DEV_best_loss')

dev_outputs_data = [rescale_factor, box_thresh, train_df, dev_dataset_for_predict, dev_pids, dev_predictions, min_box_area, boxes_by_pid_dict]
with open(os.path.join(output_dir, 'outputs_data.pkl'), 'wb') as fh:
    pickle.dump(dev_outputs_data, fh)

print('Yay!')
