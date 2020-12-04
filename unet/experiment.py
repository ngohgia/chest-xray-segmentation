import time
import os
import pickle

import torch
from torch import nn
from torch.autograd import Variable

import shutil
from metrics import *
from utilities import rescale_box_coordinates

def train(model, dataloader, optimizer, loss_fn, num_steps, boxes_by_pid_dict, rescale_factor,
          img_dim, output_dir, save_checkpoint_interval=5, min_box_area=0):
    model.train()
    
    summary = []
    average_loss = RunningAverage()
    
    average_loss_hist, loss_hist, precision_hist = [], [], []
    
    start = time.time()
    
    for i, (input_batch, labels_batch, pids_batch) in enumerate(dataloader):
        if i > num_steps:
            break
            
        input_batch = Variable(input_batch).cuda(async=True)
        labels_batch = Variable(labels_batch).cuda(async=True)
        
        
        optimizer.zero_grad()
        output_batch = model(input_batch)
       
        loss = loss_fn(output_batch, labels_batch)
        
        loss.backward()
        optimizer.step()
        
        loss_hist.append(loss.item())
        average_loss.update(loss.item())
        average_loss_hist.append(average_loss())
        
        if i % save_checkpoint_interval == 0:
            output_batch= output_batch.data.cpu().numpy()
            
            batch_precision = average_batch_precision(output_batch, pids_batch, boxes_by_pid_dict,
                                                     rescale_factor, img_dim)
            precision_hist.append(batch_precision)
            
            log = "batch loss = {:05.7f} ; ".format(loss.item())
            log += "average loss = {:05.7f} ; ".format(average_loss())
            log += "batch precision = {:05.7f} ; ".format(batch_precision)
            print('--- Train batch {} / {}: '.format(i+1, num_steps) + log)
            time_delta = time.time() - start
            print("    {:.2f} seconds".format(time_delta))
            start = time.time()
            
    metrics = "average loss = {:05.7f} ; ".format(average_loss())
    print("- Train epoch metrics: ".format(metrics))
    
    return average_loss_hist, loss_hist, precision_hist
        

def evaluate(model, dataloader, loss_fn, num_steps, boxes_by_pid_dict, rescale_factor, img_dim, min_box_area=0):
    model.eval()
    
    losses = []
    precisions = []
    
    start = time.time()
    for i, (input_batch, labels_batch, pids_batch) in enumerate(dataloader):
        if i > num_steps:
            break
        
        input_batch = Variable(input_batch).cuda(async=True)
        labels_batch = Variable(labels_batch).cuda(async=True)
        
        output_batch = model(input_batch)
        
        loss = loss_fn(output_batch, labels_batch)
        losses.append(loss.item())
        
        output_batch = output_batch.data.cpu()
        
        batch_precision = average_batch_precision(output_batch, pids_batch, boxes_by_pid_dict, rescale_factor, img_dim, return_array=True)
        for p in batch_precision:
            precisions.append(p)
        if i % 500 == 0:
            print('--- Validation batch {} / {}'.format(i+1, num_steps))

    
    mean_metrics = {'loss' : np.nanmean(losses),
                    'precision' : np.nanmean(np.asarray(precisions))}
    metrics = "average loss = {:05.7f} ; ".format(mean_metrics['loss'])
    metrics += "average precision = {:05.7f}; ".format(mean_metrics['precision'])
    print("- Eval metrics: " + metrics)
    time_delta = time.time() - start
    print(' Time {:.2f} seconds'.format(time_delta))
    
    return mean_metrics

def train_and_evaluate(model, train_dataloader, dev_dataloader, init_learning_rate, loss_fn,
                      num_epochs, num_train_steps, num_dev_steps, boxes_by_pid_dict,
                       rescale_factor, img_dim, output_dir, saved_file=None, min_box_area=0):
    if saved_file is not None:
        checkpoint = torch.load(saved_file)
        model.load_state_dict(checkpoint['model_state'])
        
    best_dev_loss = 1e+15
    best_dev_precision = 0.0
    best_loss_model = None
    best_precision_model = None
    
    train_loss_hist = []
    dev_loss_hist = []
    average_train_loss_hist = []
    train_precision_hist = []
    dev_precision_hist = []
    
    for epoch in range(num_epochs):
        start = time.time()
        
        # TODO: check this
        learning_rate = init_learning_rate * 0.5**float(epoch)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        print("Epoch {}/{}. Learning rate = {:05.3f}".format(epoch + 1, num_epochs, learning_rate))
        
        epoch_average_train_loss_hist, epoch_train_loss_hist, epoch_train_precision_hist = train(model, train_dataloader,
                optimizer, loss_fn, num_train_steps, boxes_by_pid_dict, rescale_factor, img_dim, output_dir)
        
        average_train_loss_hist += epoch_average_train_loss_hist
        train_loss_hist += epoch_train_loss_hist
        train_precision_hist += epoch_train_precision_hist
        
        
        dev_metrics = evaluate(model, dev_dataloader, loss_fn, num_dev_steps, boxes_by_pid_dict,
                              rescale_factor, img_dim)
        
        dev_loss = dev_metrics['loss']
        dev_precision = dev_metrics['precision']
        
        dev_loss_hist += len(epoch_train_loss_hist) * [dev_loss]
        dev_precision_hist += len(train_precision_hist) * [dev_precision]
        
        is_best_loss = dev_loss <= best_dev_loss
        is_best_precision = dev_precision >= best_dev_precision
        
        if is_best_loss:
            print("- New best loss: {:.4f}".format(dev_loss))
            best_dev_loss = dev_loss
            best_loss_model = model
        if is_best_precision:
            print("- New best precision: {:.4f}".format(dev_precision))
            best_dev_precision = dev_precision
            best_precision_model = model
        
        save_checkpoint({'epoch': epoch + 1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()},
                       output_dir,
                       is_best=is_best_loss,
                       metric='loss')
        save_checkpoint({'epoch': epoch + 1,
                         'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict()},
                       output_dir,
                       is_best=is_best_precision,
                       metric='precision')
        
        time_delta = time.time() - start
        print('Epoch time {:.2f} minutes'.format(time_delta / 60.))
        
    histories = {'average train loss': average_train_loss_hist,
                 'train loss': train_loss_hist,
                 'train precision': train_precision_hist,
                 'dev loss': dev_loss_hist }
    best_models = {'best loss model': best_loss_model,
                  'best precision model': best_precision_model}
   
    with open(os.path.join(output_dir, 'histories.pkl'), 'wb') as fh:
        pickle.dump(histories, fh) 
    
    return histories, best_models

def predict(model, dataloader):
    model.eval()
    
    predictions = {}
    
    for i, (test_batch, pids) in enumerate(dataloader):
        if i % 100 == 0:
            print('Predicting batch {} / {}'.format(i+1, len(dataloader)))
        
        test_batch = Variable(test_batch).cuda(async=True)
        
        sig = nn.Sigmoid().cuda()
        output_batch = model(test_batch)
        output_batch = sig(output_batch)
        output_batch = output_batch.data.cpu().numpy()
        
        for pid, output in zip(pids, output_batch):
            predictions[pid] = output
        
    return predictions

def best_box_thresh_from_dev_predictions(dev_predictions, dev_dataset_for_predict, rescale_factor, min_box_area, boxes_by_pid_dict):
    print("- Getting box threshold")
    best_threshold = None
    best_avg_dev_precision = 0.0

    thresholds = np.arange(0.1, 0.2, 0.02)
    # thresholds = np.arange(0.1, 0.2, 0.1)
    all_avg_dev_precisions = []
    for threshold in thresholds:
        print("Threshold {}".format(threshold))
        dev_precision = []
        for i in range(len(dev_dataset_for_predict)):
            if i % 500 == 0:
                print("--- Sample {} / {}".format(i+1, len(dev_dataset_for_predict)))
            img, pid = dev_dataset_for_predict[i]
            target_boxes = [rescale_box_coordinates(box, rescale_factor) for box in boxes_by_pid_dict[pid]] if pid in boxes_by_pid_dict else []
            prediction = dev_predictions[pid]
            predicted_boxes, confidences = parse_boxes(prediction, threshold=threshold, min_box_area=min_box_area, connectivity=None)
            avg_image_precision = average_image_precision(predicted_boxes, confidences, target_boxes, img_dim=img[0].shape[0])
            dev_precision.append(avg_image_precision)

        avg_dev_precision = np.nanmean(dev_precision)
        if avg_dev_precision >= best_avg_dev_precision:
            best_avg_dev_precision = avg_dev_precision
            best_threshold = threshold
    return best_threshold

def get_predicted_csv_item(predictions, pid, box_thresh, min_box_area, rescale_factor):
    prediction = predictions[pid]
    predicted_boxes, confidences = parse_boxes(prediction, threshold=box_thresh, min_box_area=min_box_area, connectivity=None)
    predicted_boxes = [rescale_box_coordinates(box, 1/rescale_factor) for box in predicted_boxes]
    return prediction_output(predicted_boxes, confidences)

def save_predictions_to_csv(test_df, test_pids, predictions, box_thresh, min_box_area, rescale_factor, output_dir, file_prefix):
    submisison_df = test_df[['patientId']].copy(deep=True)
    submisison_df['predictionString'] = submisison_df['patientId'].apply(lambda pid: get_predicted_csv_item(predictions, pid, box_thresh, min_box_area, rescale_factor) if pid in test_pids else '')

    submisison_df.to_csv(os.path.join(output_dir, file_prefix + '_submission.csv'), index=False)


def save_checkpoint(state, output_dir, is_best, metric):
    """
    Input:
        state: (dict) model's state_dict
        is_best: (bool) True if it is the best model yet
        metric: name of the metric
    """
    filename = 'last_checkpoint.pth.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, metric + '.best.pth.rar'))
