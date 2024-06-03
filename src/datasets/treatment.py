import os
import torch
import logging
import torchtext
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class MV(torch.utils.data.Dataset):
    def __init__(self, hospital, inputs, targets, scaler):
        self.identifier = hospital
        self.inputs, self.targets = inputs, targets
        self.scaler = scaler
    
    @staticmethod
    def inverse_transform(self, inputs):
        return self.scaler.inverse_transform(inputs)
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = torch.tensor(self.inputs[index]).float(), torch.tensor([self.targets[index]]).long()
        return inputs, targets
    
    def __repr__(self):
        return self.identifier

# helper method to fetch mortality prediction
def fetch_MV(args, root, seed, test_fraction):
    COL_NAME = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'targets']
    
    def _download(root):
        for hospital in URL.keys():
            _ = torchtext.utils.download_from_url(
                url=URL[hospital], 
                root=root, 
                hash_value=MD5[hospital], 
                hash_type='md5'
            )
            os.rename(os.path.join(root, URL[hospital].split('/')[-1]), os.path.join(root, f'MV ({hospital}).csv'))
    
    def _munge_and_split(root, hospital, seed, test_fraction):
        # load data
        to_drop = [
            10, # the slope of the peak exercise ST segment
            11, # number of major vessels (0-3) colored by flourosopy
            12 # thalassemia background
        ]
        df = pd.read_csv(os.path.join(root, f'MV ({hospital}).csv'), header=None, na_values='?', usecols=[i for i in range(14) if i not in to_drop]).apply(lambda x: x.fillna(x.mean()),axis=0).reset_index(drop=True)
        
        # rename column
        df.columns = COL_NAME
        
        # adjust dtypes
        df['targets'] = df['targets'].where(df['targets'] == 0, 1)
        df['age'] = df['age'].astype(float)
        df['sex'] = df['sex'].astype(int)
        df['cp'] = df['cp'].astype(int)
        df['trestbps'] = df['trestbps'].astype(float)
        df['chol'] = df['chol'].astype(float)
        df['restecg'] = df['restecg'].astype(int)
        df['cp'] = df['cp'].astype(int)
        df['thalach'] = df['thalach'].astype(float)
        df['exang'] = df['exang'].astype(int)
        df['oldpeak'] = df['oldpeak'].astype(float)
        
        # get one-hot encoded dummy columns for categorical data
        df = pd.concat([pd.get_dummies(df.iloc[:, :-1], columns=['cp', 'restecg'], drop_first=True, dtype=int), df[['targets']]], axis=1)
        
        # get inputs and targets
        inputs, targets = df.iloc[:, :-1].values, df.iloc[:, -1].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_fraction, random_state=seed, stratify=targets)
        
        # scaling inputs
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)
        return (
            MV(f'[MV] CLIENT < {hospital} > (train)', train_inputs, train_targets, scaler), 
            MV(f'[MV] CLIENT < {hospital} > (test)', test_inputs, test_targets, scaler)
        ) 
        
    logger.info(f'[LOAD] [MV] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(os.path.join(root, 'MV')):
        _download(root=os.path.join(root, 'MV'))
        logger.info(f'[LOAD] [MV] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [MV] ...raw data already exists!')
    
    logger.info(f'[LOAD] [MV] Munging and splitting dataset!')
    client_datasets = []
    for hospital in URL.keys():
        client_datasets.append(_munge_and_split(os.path.join(root, 'MV'), hospital, seed, test_fraction))
    logger.info('[LOAD] [MV] ...munged and splitted dataset!')
    
    args.in_features = 37
    args.num_classes = 2
    args.K = 26
    return {}, client_datasets, args
