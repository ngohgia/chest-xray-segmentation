import pandas as pd

input_file = '/home/ghn8/pneumonia-detection/output/UNET_SingleView_Baseline/1115-115307/best_loss_submission.csv'
orig_df = pd.read_csv(input_file)
orig_df.drop_duplicates(subset ="patientId", keep = False, inplace = True) 

output_file = '/home/ghn8/pneumonia-detection/output/UNET_SingleView_Baseline/1115-115307/best_loss_submission_UNDUP.csv'
orig_df.to_csv(output_file, index=False)
