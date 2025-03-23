import cv2
import torch
import argparse
import numpy as np
import albumentations as A
import torchvision.transforms as transforms

from torch import nn
from PIL import Image
from torch import autocast
from einops import rearrange
from omegaconf import OmegaConf
from contextlib import nullcontext
from torchvision.transforms import Resize
from ldm.models.diffusion.ddim import DDIMSampler
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model, faceParsing_demo
from our_work.scripts.utils import crop_and_align_face, get_tensor, get_tensor_clip, load_model_from_config, un_norm_clip, un_norm

class REFace():

    def __init__(
            self,
            device,
            config_path: str = "models/REFace/configs/project.yaml",
            ckpt_path: str = "models/REFace/checkpoints/last.ckpt",
            faceParser_name: str = "default",
            faceParsing_ckpt: str = "Other_dependencies/face_parsing/79999_iter.pth",
            segnext_config: str = "",
            ddim_steps: int = 50,
            dlib_landmark_path: str = "Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat"
        ):
        
        self.device = device
        self.faceparsing_model = None
        self.precision = "autocast"
        self.Fullmask = False

        self.C = 4 # latent chanels
        self.H = 512 # image height, in pixel space
        self.f = 8 # downsampling factor
        self.W = 512 # image width, in pixel space

        self.ddim_steps = ddim_steps
        self.ddim_eta = 0
        self.seg12 = True

        self.scale = 3.5

        self.faceParser_name = faceParser_name
        self.config = OmegaConf.load(config_path)
        self.args = self.config.data.params.test.params

        self.preserve=self.args['preserve_mask_src']

        self.model = load_model_from_config(self.config, ckpt_path, verbose=False)
        self.faceParsing_model = init_faceParsing_pretrained_model(self.faceParser_name, faceParsing_ckpt, segnext_config)

        self.model.to(self.device)

        self.sampler = DDIMSampler(self.model)

        self.dlib_landmark_path = dlib_landmark_path

    def process_source_image(self, img: Image.Image) -> torch.Tensor:
        
        img_mask, img_cropped, _ = self.compute_mask_and_cropped(img)
        
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

        return ref_image_tensor

    def process_target_image(self, img: Image.Image) -> tuple[torch.Tensor, dict, torch.Tensor]:
        
        mask_img, cropped_img, inv_transforms = self.compute_mask_and_cropped(img)

        if mask_img is None:
            return None, None, None
        
        mask_img = mask_img.convert('L')
        img_p = Image.fromarray(cropped_img).convert('RGB').resize((512,512))
        
        
        if self.Fullmask:
            mask_img_full=mask_img
            mask_img_full=get_tensor(normalize=False, toTensor=True)(mask_img_full)
        
        mask_img = np.array(mask_img)  # Convert the label to a NumPy array if it's not already

        # Create a mask to preserve values in the 'preserve' list
        preserve=self.preserve
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

    def compute_mask_and_cropped(self, img: Image.Image) -> tuple[Image.Image, np.ndarray]:

        try:
            crops, _, _, inv_transforms = crop_and_align_face([img], self.dlib_landmark_path)
            crops = [crop.convert("RGB") for crop in crops]
            T = crops[0]
            
            pil_im = T.resize((1024,1024), Image.BILINEAR)
            mask = faceParsing_demo(self.faceParsing_model, pil_im, convert_to_seg12=self.seg12, model_name=self.faceParser_name)
            mask = Image.fromarray(mask)
            cropped = np.array(T)

        except Exception as e:
            print(f"Error in compute_mask_and_cropped() : {e}")
            return None, None, None

        return mask, cropped, inv_transforms

    def generate(
            self,
            source_img: torch.Tensor,
            target_img: torch.Tensor,
            test_model_kwargs: dict[str, torch.Tensor],
            inv_transforms: torch.Tensor,
            orig_target: Image.Image
        ) -> Image.Image:
        
        precision_scope = autocast if self.precision=="autocast" else nullcontext

        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():

                    source_img = source_img.unsqueeze(0)
                    target_img = target_img.unsqueeze(0)
                    inv_transforms = [inv_transforms]
                    
                    test_model_kwargs={n:test_model_kwargs[n].unsqueeze(0).to(self.device, non_blocking=True) for n in test_model_kwargs}
                    uc = None
                    if self.scale != 1.0:
                        uc = self.model.learnable_vector.repeat(target_img.shape[0],1,1)
                        if self.model.stack_feat:
                            uc2 = self.model.other_learnable_vector.repeat(target_img.shape[0],1,1)
                            uc = torch.cat([uc,uc2],dim=-1)
                    
                    landmarks = self.model.get_landmarks(target_img) if self.model.Landmark_cond else None
                    ref_imgs= source_img.to(self.device, non_blocking=True)
                    
                    c = self.model.conditioning_with_feat(ref_imgs.squeeze(1).to(torch.float32),landmarks=landmarks,tar=target_img.to(self.device).to(torch.float32)).float()
                    if (self.model.land_mark_id_seperate_layers or self.model.sep_head_att) and self.scale != 1.0:
            
                        # concat c, landmarks
                        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                        uc=torch.cat([uc,landmarks],dim=-1)
                    
                    
                    if c.shape[-1]==1024:
                        c = self.model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)

                    z_inpaint = self.model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = self.model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [self.C, self.H // self.f, self.W // self.f]

                    samples_ddim, _ = self.sampler.sample(
                        S=self.ddim_steps,
                        conditioning=c,
                        batch_size=target_img.shape[0],
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=self.scale,
                        unconditional_conditioning=uc,
                        eta=self.ddim_eta,
                        test_model_kwargs=test_model_kwargs,
                        src_im=ref_imgs.squeeze(1).to(torch.float32),
                        tar=target_img.to(self.device)
                    )

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    sample = x_checked_image_torch[0]
                    sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')

                    img = Image.fromarray(sample.astype(np.uint8)).resize((1024,1024), Image.BILINEAR)

                    inv_trans_coeffs = inv_transforms[0][0].tolist()
                    swapped_and_pasted = img.convert('RGBA')
                    pasted_image = orig_target.convert('RGBA')
                    swapped_and_pasted.putalpha(255)
                    # breakpoint()
                    projected = swapped_and_pasted.transform(orig_target.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
                    pasted_image.alpha_composite(projected)
                    #remove alpha channel
                    final_image = pasted_image.convert('RGB')

        return final_image

    def face_swapp(self, source: Image.Image, target: Image.Image):

        # 0:background, 1:lip, 2:eyebrows, 3:eyes, 4:hair, 5:nose, 6:skin, 7:ears, 
        # 8:belowface, 9:mouth, 10:eye_glass, 11:ear_rings
        self.preserve = [1,2,3,5,6,7,9]

        source_tensor = self.process_source_image(source)
        target_tensor, test_model_kwargs, inv_transforms = self.process_target_image(target)

        return self.generate(
            source_img=source_tensor,
            target_img=target_tensor,
            test_model_kwargs=test_model_kwargs,
            inv_transforms=inv_transforms,
            orig_target=target
        )

    def head_swapp(self, source: Image.Image, target: Image.Image):

        # 0:background, 1:lip, 2:eyebrows, 3:eyes, 4:hair, 5:nose, 6:skin, 7:ears, 
        # 8:belowface, 9:mouth, 10:eye_glass, 11:ear_rings
        self.preserve = [1,2,3, 4, 5,6,7,9]

        source_tensor = self.process_source_image(source)
        target_tensor, test_model_kwargs, inv_transforms = self.process_target_image(target)

        return self.generate(
            source_img=source_tensor,
            target_img=target_tensor,
            test_model_kwargs=test_model_kwargs,
            inv_transforms=inv_transforms,
            orig_target=target
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('source', help='image source for the swapping.')
    parser.add_argument('target', help='image source for the swapping.')
    parser.add_argument('out_path', help='file path to store the output.')
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/REFace/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument("--head_swapp", action='store_true')


    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    source = Image.open(args.source).convert("RGB")
    target = Image.open(args.target).convert("RGB")

    model = REFace(device, args.config, args.ckpt, ddim_steps=args.ddim_steps)

    if args.head_swapp:
        out = model.head_swapp(source, target)
    else:
        out = model.face_swapp(source, target)

    out.save(args.out_path)

    print(f"Your image is ready at path : {args.out_path}")