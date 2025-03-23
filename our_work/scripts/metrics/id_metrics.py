import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
from PIL import Image
import re
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from src.Face_models.encoders.model_irse import Backbone
# import clip
import torchvision


try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x
import cv2
import albumentations as A
import torch.nn as nn
from natsort import natsorted

from datasets import load_dataset, Dataset

def un_norm_clip(x1):
    x = x1*1.0 # to avoid changing the original tensor or clone() can be used
    reduce=False
    if len(x.shape)==3:
        x = x.unsqueeze(0)
        reduce=True
    x[:,0,:,:] = x[:,0,:,:] * 0.26862954 + 0.48145466
    x[:,1,:,:] = x[:,1,:,:] * 0.26130258 + 0.4578275
    x[:,2,:,:] = x[:,2,:,:] * 0.27577711 + 0.40821073
    
    if reduce:
        x = x.squeeze(0)
    return x

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


class IDDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            celeba_img_path: str,
            celeba_mask_path: str,
            transforms=None,
            data_name: str = "celeba",
            split: str = "train", # train (or test)
        ):
        self.hf_dataset = hf_dataset[split]
        self.celeba_img_path = celeba_img_path
        self.celeba_mask_path = celeba_mask_path
        self.transforms = transforms
        self.data_name=data_name

        self.trans=A.Compose([
            A.Resize(height=112,width=112)]
        )
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        id_source = self.hf_dataset[i]["id_source"]

        img_swapped: Image.Image = self.hf_dataset[i]["image"]
        img_swapped = img_swapped.convert('RGB')
        img_source = Image.open(os.path.join(self.celeba_img_path, id_source+".jpg")).convert('RGB')

        ref_mask_img = Image.open(os.path.join(self.celeba_mask_path, id_source+".png")).convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        if self.data_name=="celeba":
            preserve = [1,2,4,5,8,9 ,6,7,10,11,12 ]
        else:
            preserve=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]  # No mask

        ref_mask= np.isin(ref_mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        
        reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        mask_ref=TF.Resize((112,112))(reference_mask_tensor)

        # swapped image

        ref_img=self.trans(image=np.array(img_swapped))
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor()(ref_img)
        ref_img=ref_img*mask_ref
        processed_img_swapped = ref_img.unsqueeze(0)

        # source image

        ref_img=self.trans(image=np.array(img_source))
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor()(ref_img)
        ref_img=ref_img*mask_ref
        processed_img_source = ref_img.unsqueeze(0)
        
        return processed_img_swapped, processed_img_source

class IDLoss(nn.Module):
    def __init__(self,multiscale=True):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        # self.opts = opts 
        self.multiscale = multiscale
        self.face_pool_1 = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
    
        self.facenet.load_state_dict(torch.load("Other_dependencies/arcface/model_ir_se50.pth"))
        
        self.face_pool_2 = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        
        self.set_requires_grad(False)
            
    def set_requires_grad(self, flag=True):
        for p in self.parameters():
            p.requires_grad = flag
    
    def extract_feats(self, x,clip_img=False):
        # breakpoint()
        if clip_img:
            x = un_norm_clip(x)
            x = TFF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = self.face_pool_1(x)  if x.shape[2]!=256 else  x # (1) resize to 256 if needed
        x = x[:, :, 35:223, 32:220]  # (2) Crop interesting region
        x = self.face_pool_2(x) # (3) resize to 112 to fit pre-trained model
        # breakpoint()
        x_feats = self.facenet(x, multi_scale=self.multiscale )

        return x_feats

    def forward(self, x,clip_img=False):
        x_feats_ms = self.extract_feats(x,clip_img=clip_img)
        return x_feats_ms[-1]
   

class IDMetric():
    def __init__(self,
        device: str = "cpu",
        batch_size: int = 20,
        num_workers: int = 1,
    ):
        
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load Model
        self.model = IDLoss().to(device)
        self.model.eval()

    def compute(
            self,
            hf_dataset: Dataset,
            local_celeba_img_path: str,
            local_celeba_mask_path: str,
            model: str=None,
        ):

        if model is not None:
            hf_dataset = hf_dataset.filter(lambda x: x["model"]==model)

        dataset = IDDataset(
            hf_dataset,
            local_celeba_img_path,
            local_celeba_mask_path,
            transforms=TF.ToTensor()
        )
        
        swapped_pred_arr, source_pred_arr = self.compute_features_wrapp(dataset)

        # find l2 distance
        dist = np.linalg.norm(swapped_pred_arr-source_pred_arr,axis=1)
        Value= np.mean(dist)
        
        return Value

    def compute_features_wrapp(self, dataset: IDDataset):

        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- dataset     : dataset object on which compute features
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        self.model.eval()

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=self.num_workers)


        swapped_pred_arr = np.empty((len(dataset), 512))
        source_pred_arr = np.empty((len(dataset), 512))

        swapped_start_idx = 0
        source_start_idx = 0

        for batch_img_swapped, batch_img_source in tqdm(dataloader):

            batch_img_swapped = batch_img_swapped.to(self.device).squeeze(1)
            batch_img_source = batch_img_source.to(self.device).squeeze(1)

            swapped_pred_arr, swapped_start_idx = self.compute_features(batch_img_swapped, swapped_pred_arr, swapped_start_idx)
            source_pred_arr, source_start_idx = self.compute_features(batch_img_source, source_pred_arr, source_start_idx)
            
        return swapped_pred_arr, source_pred_arr
    
    def compute_features(self, batch, pred_arr: np.ndarray, start_idx: int):

        with torch.no_grad():
            pred = self.model(batch)

        pred = pred.cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

        return pred_arr, start_idx


if __name__ == '__main__':
    
    metric = IDMetric(
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    hf_identifier = "florian-morel22/deepfake-celebAMask-HQ-REFace"
    hf_token = os.getenv("HF_TOKEN")
    local_celeba_img_path = "datasets/CelebAMask-HQ/CelebA-HQ-img"
    local_celeba_mask_path = "datasets/CelebAMask-HQ/CelebA-HQ-mask"
    model="REFace"

    # Load dataset
    hf_dataset = load_dataset(hf_identifier, token=hf_token)

    result = metric.compute(
        hf_dataset,
        local_celeba_img_path,
        local_celeba_mask_path,
        model
    )

    print("ID metric : ", result)