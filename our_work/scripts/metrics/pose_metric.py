import os

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
import torch.nn.functional as F
import eval_tool.face_vid2vid.modules.hopenet as hopenet1
from torchvision import models

from datasets import load_dataset, Dataset

import torchvision
from tqdm import tqdm
from dotenv import load_dotenv
# from inception import InceptionV3

load_dotenv()


class PoseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            celeba_path: str,
            transforms=None,
            split: str = "train", # train or test
        ):
        self.hf_dataset = hf_dataset[split]
        self.celeba_path = celeba_path
        self.transforms = transforms

        self.transform_hopenet =  torchvision.transforms.Compose([TF.ToTensor(),TF.Resize(size=(224, 224)),
                                                     TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, i):
        
        id_target = self.hf_dataset[i]["id_target"]

        img_swapped = self.hf_dataset[i]["image"]
        img_target = Image.open(os.path.join(self.celeba_path, id_target+".jpg")).convert('RGB')

        img_swapped = self.transform_hopenet(img_swapped)
        img_target = self.transform_hopenet(img_target)

        return img_swapped, img_target

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

class PoseMetric():
    def __init__(self,
        local_hopenet_path: str = "Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl",
        device: str = "cpu",
        batch_size: int = 20,
        num_workers: int = 1,
    ):
        
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load Model
        self.model = hopenet1.Hopenet(models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        print('Loading hopenet')
        hopenet_state_dict = torch.load(local_hopenet_path)
        self.model.load_state_dict(hopenet_state_dict)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model.eval()

    def compute(
            self,
            hf_dataset: Dataset,
            local_celeba_path: str,
            model: str,
        ):

        hf_dataset = hf_dataset.filter(lambda x: x["model"]==model)
        local_celeba_path = local_celeba_path

        dataset = PoseDataset(
            hf_dataset,
            local_celeba_path,
            transforms=TF.ToTensor()
        )
        
        swapped_pred_arr, target_pred_arr = self.compute_features_wrapp(dataset)

        # find l2 distance
        dist = np.linalg.norm(swapped_pred_arr-target_pred_arr,axis=1)
        Value= np.mean(dist)
        
        return Value

    def compute_features_wrapp(self, dataset: PoseDataset):

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


        swapped_pred_arr = np.empty((len(dataset),3))
        swapped_start_idx = 0

        target_pred_arr = np.empty((len(dataset),3))
        target_start_idx = 0

        for batch_img_swapped, batch_img_target in tqdm(dataloader):

            batch_img_swapped = batch_img_swapped.to(self.device)
            batch_img_target = batch_img_target.to(self.device)

            swapped_pred_arr, swapped_start_idx = self.compute_features(batch_img_swapped, swapped_pred_arr, swapped_start_idx)
            target_pred_arr, target_start_idx = self.compute_features(batch_img_target, target_pred_arr, target_start_idx)

        return swapped_pred_arr, target_pred_arr
    
    def compute_features(self, batch, pred_arr, start_idx: int):

        with torch.no_grad():

            yaw_gt, pitch_gt, roll_gt = self.model(batch)
            yaw_gt = headpose_pred_to_degree(yaw_gt)
            pitch_gt = headpose_pred_to_degree(pitch_gt)
            roll_gt = headpose_pred_to_degree(roll_gt)

        yaw_gt = yaw_gt.cpu().numpy().reshape(-1,1)
        pitch_gt = pitch_gt.cpu().numpy().reshape(-1,1)
        roll_gt = roll_gt.cpu().numpy().reshape(-1,1)

        pred_arr[start_idx:start_idx + yaw_gt.shape[0]] = np.concatenate((yaw_gt,pitch_gt,roll_gt),axis=1)

        start_idx = start_idx + yaw_gt.shape[0]

        return pred_arr, start_idx
    
if __name__ == '__main__':
    
    metric = PoseMetric(
        local_hopenet_path="Other_dependencies/Hopenet_pose/hopenet_robust_alpha1.pkl",
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    hf_identifier = "florian-morel22/deepfake-real"
    hf_token = os.getenv("HF_TOKEN")
    local_celeba_path = "dataset/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
    model="Real"

    # Load dataset
    hf_dataset = load_dataset(hf_identifier, token=hf_token)

    pose = metric.compute(
        hf_dataset,
        local_celeba_path,
        "Real"
    )

    print("position metric : ", pose)