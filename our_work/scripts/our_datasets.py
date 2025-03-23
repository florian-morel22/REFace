import os 
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
import albumentations as A
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from os import path as osp
from src.utils.alignmengt import crop_faces, calc_alignment_coefficients, crop_faces_from_image
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo

from our_work.scripts.utils import get_tensor, crop_and_align_face, get_tensor_clip


class OurCelebADataset(data.Dataset):
    def __init__(
            self,
            faceParsing_model,
            faceParser_name,
            seg12,
            data_path: str,
            splitted_celeba_path: str,
            model_name: str=None,
            label_transform=None,
            **args
        ):

        self.label_transform=label_transform
        self.args=args
        self.load_prior=False
        self.kernel = np.ones((1, 1), np.uint8)
        self.gray_outer_mask=True
        
        self.Fullmask=False
        self.bbox_path_list=[]
        
        self.data_path=data_path
        self.gray_outer_mask=args['gray_outer_mask']

        self.faceParsing_model = faceParsing_model
        self.faceParser_name = faceParser_name
        self.seg12 = seg12

        if hasattr(args, 'preserve_mask'):
            self.preserve=args['preserve_mask']
            self.remove_tar=args['preserve_mask']
            self.preserve_src=args['preserve_mask']
        else:
            self.preserve=args['remove_mask_tar_FFHQ']
            self.remove_tar=args['remove_mask_tar_FFHQ']
            self.preserve_src=args['preserve_mask_src_FFHQ']
    
        self.Fullmask=False

        splitted_celeba = pd.read_csv(splitted_celeba_path, dtype={"source": str, "target": str})
        if model_name is not None:
            splitted_celeba = splitted_celeba[splitted_celeba['model']==model_name]

        start = args["start"] if "start" in args else 0
        stop = args["stop"] if "stop" in args else len(splitted_celeba)
        splitted_celeba = splitted_celeba[start:stop]
        splitted_celeba.reset_index(drop=True, inplace=True)


        self.imgs = {
            i:{
                "source_id": row['source'],
                "target_id": row['target'],
                "source_path": osp.join(data_path, row['source'] + ".jpg"),
                "target_path": osp.join(data_path, row['target'] + ".jpg"),
            }
            for i,  row in splitted_celeba.iterrows()
        }

        # image pairs indices
        self.indices = np.arange(len(self.imgs))
        self.length=len(self.indices)
        # self.preserve=preserve

    def __getitem__(self, index):
        if self.gray_outer_mask:
            return self.__getitem_gray__(index)
        else:
            return self.__getitem_black__(index) #buseless

    def __getitem_gray__(self, index):
        # uses the black mask in reference

        # index = self.start + index

        target_id = self.imgs[index]["target_id"]
        source_id = self.imgs[index]["source_id"]
        target_path = self.imgs[index]["target_path"]
        target_tensor = self.imgs[index]["target_tensor"]
        source_tensor = self.imgs[index]["source_tensor"]
        test_model_kwargs = self.imgs[index]["test_model_kwargs"]
        inv_transforms = self.imgs[index]["inv_transforms"]

        segment_id_batch = str(index)

        if target_tensor is None or source_tensor is None:
            return torch.empty(1), torch.empty(1), {}, segment_id_batch, torch.empty(1), "ERROR"

        return target_tensor, source_tensor, test_model_kwargs, segment_id_batch, inv_transforms, target_path, target_id, source_id



    def __getitem_black__(self, index):
        # uses the black mask in reference

        # Useless for dataset inference
        pass

    def __len__(self):
        return self.length
    
    def process_data(self):
        
        for index in tqdm(self.imgs.keys()):
            source_path = self.imgs[index]["source_path"]
            target_path = self.imgs[index]["target_path"]

            target_tensor, test_model_kwargs, inv_transforms = self.process_target_image(target_path)
            source_tensor = self.process_source_image(source_path)

            self.imgs[index]["target_tensor"] = target_tensor
            self.imgs[index]["source_tensor"] = source_tensor
            self.imgs[index]["test_model_kwargs"] = test_model_kwargs
            self.imgs[index]["inv_transforms"] = inv_transforms

    def compute_mask_and_cropped(self, img_path: str) -> tuple[Image.Image, np.ndarray]:

        try:
            crops, _, _, inv_transforms = crop_and_align_face([img_path])
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
            
            pil_im = T.resize((1024,1024), Image.BILINEAR)
            mask = faceParsing_demo(self.faceParsing_model, pil_im, convert_to_seg12=self.seg12, model_name=self.faceParser_name)
            mask = Image.fromarray(mask)
            cropped = np.array(T)

        except Exception as e:
            print(f"Error in {img_path} : {e}")
            return None, None, None

        return mask, cropped, inv_transforms

    def process_source_image(self, img_path: str) -> torch.Tensor:

        img_mask, img_cropped, _ = self.compute_mask_and_cropped(img_path)
        
        if img_mask is None:
            return None

        trans=A.Compose([
                A.Resize(height=224,width=224)
        ])

        ref_img = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
        
        ref_mask_img = img_mask.convert('L')
        ref_mask_img = np.array(ref_mask_img)  # Convert the label to a NumPy array if it's not already

        ref_mask= np.isin(ref_mask_img, self.args["preserve_mask_src"])

        # Create a converted_mask where preserved values are set to 255
        ref_converted_mask = np.zeros_like(ref_mask_img)
        ref_converted_mask[ref_mask] = 255
        ref_converted_mask=Image.fromarray(ref_converted_mask).convert('L')
        # convert to PIL image
        reference_mask_tensor=get_tensor(normalize=False, toTensor=True)(ref_converted_mask)
        mask_ref=transforms.Resize((224,224))(reference_mask_tensor)
        ref_img=trans(image=ref_img)
        ref_img=Image.fromarray(ref_img["image"])
        ref_img=get_tensor_clip()(ref_img)
        ref_img=ref_img*mask_ref
        ref_image_tensor = ref_img.to(torch.float16).unsqueeze(0)

        return ref_image_tensor # .to(device,non_blocking=True)
    
    def process_target_image(self, img_path: str):
        
        mask_img, cropped_img, inv_transforms = self.compute_mask_and_cropped(img_path)

        if mask_img is None:
            return None, None, None
        
        mask_img = mask_img.convert('L')
        img_p = Image.fromarray(cropped_img).convert('RGB').resize((512,512))
        
        
        if self.Fullmask:
            mask_img_full=mask_img
            mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        # preserve = [1,2,4,5,8,9,17 ]
        # preserve = [1,2,4,5,8,9, 6,7,10,11,12]
        preserve=self.remove_tar
        # preserve=[2,3,5,6,7] 
        mask = np.isin(mask_img, preserve)

        # Create a converted_mask where preserved values are set to 255
        converted_mask = np.zeros_like(mask_img)
        converted_mask[mask] = 255
        # convert to PIL image
        mask_img=Image.fromarray(converted_mask).convert('L') 


        ### Crop input image
        image_tensor = get_tensor()(img_p)
        W,H = img_p.size

   

        mask_tensor=1-get_tensor(normalize=False, toTensor=True)(mask_img)

        inpaint_tensor=image_tensor*mask_tensor

        return image_tensor, {"inpaint_image":inpaint_tensor,"inpaint_mask":mask_tensor}, inv_transforms