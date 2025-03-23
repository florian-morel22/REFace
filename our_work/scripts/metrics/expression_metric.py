import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from eval_tool.Deep3DFaceRecon_pytorch_edit.options.test_options import TestOptions
from eval_tool.Deep3DFaceRecon_pytorch_edit.models import create_model

# give empty string to use the default options
test_opt = TestOptions('')
test_opt = test_opt.parse()

load_dotenv()


class ExpressionDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hf_dataset: Dataset,
            celeba_path: str,
            split: str = "train", # train or test
        ):
        self.hf_dataset = hf_dataset[split]
        self.celeba_path = celeba_path
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, i):

        id_target = self.hf_dataset[i]["id_target"]

        img_swapped: Image.Image = self.hf_dataset[i]["image"]
        img_swapped = img_swapped.convert('RGB')
        img_target = Image.open(os.path.join(self.celeba_path, id_target+".jpg")).convert('RGB')

        img_swapped = img_swapped.resize((512, 512), Image.BICUBIC)
        img_target = img_target.resize((512, 512), Image.BICUBIC)
        
        img_swapped = torch.tensor(np.array(img_swapped)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_target = torch.tensor(np.array(img_target)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        return img_swapped, img_target

class ExpressionMetric():
    def __init__(self,
        device: str = "cpu",
        batch_size: int = 20,
        num_workers: int = 1,
    ):
        
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load Model
        self.model = create_model(test_opt)
        self.model.setup(test_opt)
        
        if torch.cuda.is_available():
            self.model.net_recon.cuda()
            self.model.facemodel.to("cuda")
        self.model.eval()

    def compute(
            self,
            hf_dataset: Dataset,
            local_celeba_path: str,
            model: str=None,
        ):

        if model is not None:
            hf_dataset = hf_dataset.filter(lambda x: x["model"]==model)

        dataset = ExpressionDataset(
            hf_dataset,
            local_celeba_path,
        )
        
        swapped_pred_arr, target_pred_arr = self.compute_features_wrapp(dataset)

        # find l2 distance
        diff_feat=np.power(swapped_pred_arr-target_pred_arr,2)
        
        # diff_feat = np.abs(feat1-feat2)
        diff_feat = np.sum(diff_feat,axis = -1)
        Value = np.sqrt(diff_feat)
        Value = np.mean(Value)
        
        return Value

    def compute_features_wrapp(self, dataset: ExpressionDataset):

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


        swapped_pred_arr = np.empty((len(dataset),64))
        swapped_start_idx = 0

        target_pred_arr = np.empty((len(dataset),64))
        target_start_idx = 0

        for batch_img_swapped, batch_img_target in tqdm(dataloader):

            batch_img_swapped = batch_img_swapped.to(self.device).squeeze(1)
            batch_img_target = batch_img_target.to(self.device).squeeze(1)

            swapped_pred_arr, swapped_start_idx = self.compute_features(batch_img_swapped, swapped_pred_arr, swapped_start_idx)
            target_pred_arr, target_start_idx = self.compute_features(batch_img_target, target_pred_arr, target_start_idx)

        return swapped_pred_arr, target_pred_arr
    
    def compute_features(self, batch, pred_arr, start_idx: int):

        with torch.no_grad():

            coeff = self.model.forward(batch)
            pred = coeff['exp']

        pred = pred.cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

        return pred_arr, start_idx
    
if __name__ == '__main__':
    
    metric = ExpressionMetric(
        device="cuda",
        batch_size=20,
        num_workers=1,
    )

    hf_identifier = "florian-morel22/deepfake-REFace"
    hf_token = os.getenv("HF_TOKEN")
    local_celeba_path = "dataset/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"
    model="REFace"

    # Load dataset
    hf_dataset = load_dataset(hf_identifier, token=hf_token)

    expression = metric.compute(
        hf_dataset,
        local_celeba_path,
        model
    )

    print("expression metric : ", expression)