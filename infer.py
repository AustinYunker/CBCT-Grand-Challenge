
import torch, os, sys, logging
from data import CBCT_Test_Dataset
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from model import unet

#Specify the data location and dosage level

test_dir = '/eagle/AI-based-NDI-Spirit/Loyola/GC/test'
#dosage level: either clinical_dose or low_dose
dosage_level = 'clinical_dose'

#logging info
logging.basicConfig(filename='GrandChallenge.log', level=logging.DEBUG,\
                    format='%(asctime)s %(levelname)s %(module)s: %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

#load the data
logging.info("\nLoading data into CPU memory, it will take a while ... ...")

#Dataloader
ds_test = CBCT_Test_Dataset(ifn=test_dir, dosage_level=dosage_level)
dl_test = DataLoader(dataset=ds_test, batch_size=1, shuffle=False,\
                        num_workers=4, prefetch_factor=1, drop_last=False, pin_memory=True)

logging.info(f"\nLoaded %d samples, {ds_test.dim}, into CPU memory for training." % (len(ds_test), ))

#CPU/GPU device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load in model based on dosage level
if dosage_level == 'clinical_dose':
    logging.info('\nRunning inference using clinical dose model')
    mdl_pth = 'Models/CD/unet.pth'
else:
    logging.info('\nRunning inference using low dose model')
    mdl_pth = 'Models/LD/unet.pth'

checkpoint = torch.load(mdl_pth, map_location=torch.device('cpu'))
model = unet(start_filter_size=4)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(dev)

#Loop through test data and collect test mse vals
mse_res = []
model.eval()
with torch.no_grad():
    #(noisy_reconstruction, target_reconstruction)
    for X_mb, y_mb in dl_test:
        X_mb_dev = X_mb.to(dev)
        pred = model(X_mb_dev).cpu().squeeze()#
        y_mb = y_mb.squeeze()

        mse = torch.nn.functional.mse_loss(pred, y_mb).numpy()
        mse_res.append(mse)

mse_res = np.array(mse_res)
#put test results in dataframe
mse_df = pd.DataFrame({
    'MSE': mse_res
})

#print test statistics
logging.info('\nTest Results: \n')
logging.info(mse_df.describe())

#save test results to csv file with dosage level naming convention
mse_df.to_csv(f'{dosage_level}_mse_test_results.csv', columns=mse_df.columns, index=False)

