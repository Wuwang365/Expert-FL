from torch.utils.data import Dataset
import glob
from PIL import Image
import numpy as np
import torch

class TargetDataset(Dataset):
    def __init__(self,data_path,shot) -> None:
        super().__init__()
        self.data = []
        self.shot = shot
        self.data_path = f'{data_path}/{self.shot}/*.png'
        self.load_data()
        
        
    def load_data(self):
        names = glob.glob(self.data_path)
        for name in names:
            img = Image.open(name)
            img = np.asarray(img)/255.0
            img = torch.as_tensor(img,dtype=torch.float)
            img = img.permute(2,0,1)
            self.data.append(img)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
class NonTargetDataset(Dataset):
    def __init__(self,data_path,shot) -> None:
        super().__init__()
        self.data = []
        self.shot = shot
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        path = f'{self.data_path}/*/*.png'
        target_path = f'{self.data_path}/{self.shot}/*.png'
        target_names = glob.glob(target_path)
        names = glob.glob(path)
        for i in target_names:
            names.remove(i)
        for name in names:
            img = Image.open(name)
            img = np.asarray(img)/255.0
            img = torch.as_tensor(img,dtype=torch.float)
            img = img.permute(2,0,1)
            self.data.append(img)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
class VoteDataset(Dataset):
    def __init__(self,path) -> None:
        super().__init__()
        self.data = []
        self.load_data(path=path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0],self.data[index][1]
    
    def load_data(self,path):
        names = glob.glob(f'{path}/*/*.png')
        for name in names:
            label = int(name.split('/')[-2])
            label = torch.tensor(label)
            img = Image.open(name)
            img = np.asarray(img)/255.0
            img = torch.as_tensor(img,dtype=torch.float)
            img = img.permute(2,0,1)
            self.data.append((label,img))