import torch
import numpy as np
import argparse, os

from PIL import Image
from torch import autocast
from einops import rearrange
from itertools import islice
from omegaconf import OmegaConf
from contextlib import nullcontext
from torchvision.utils import make_grid
from torchvision.transforms import Resize
from ldm.util import instantiate_from_config
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from pretrained.face_parsing.face_parsing_demo import init_faceParsing_pretrained_model

from our_work.scripts.our_datasets import OurCelebADataset

from push_dataset_to_hub import update_dataset


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a photograph of an astronaut riding a horse",
    help="the prompt to render"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="results_video/debug"
)
parser.add_argument(
    "--Base_dir",
    type=str,
    nargs="?",
    help="dir to write cropped_images",
    default="results_video"
)
parser.add_argument(
    "--skip_grid",
    action='store_true',
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
    default=False
)
parser.add_argument(
    "--Start_from_target",
    action='store_true',
    help="if enabled, uses the noised target image as the starting ",
)
parser.add_argument(
    "--only_target_crop",
    action='store_true',
    help="if enabled, uses the noised target image as the starting ",
    default=True
)
parser.add_argument(
    "--target_start_noise_t",
    type=int,
    default=1000,
    help="target_start_noise_t",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=2,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=12,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--data_folder",
    type=str,
    help="folder with all images",
    default="examples/faceswap",
)
parser.add_argument(
    "--src_image_mask",
    type=str,
    help="src_image_mask",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
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
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--rank",
    type=int,
    default=0,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

parser.add_argument('--faceParser_name', default='default', type=str, help='face parser name, [ default | segnext] is currently supported.')
parser.add_argument('--faceParsing_ckpt', type=str, default="Other_dependencies/face_parsing/79999_iter.pth")  
parser.add_argument('--segnext_config', default='', type=str, help='Path to pre-trained SegNeXt faceParser configuration file, '
                                                                    'this option is valid when --faceParsing_ckpt=segenext')
        
parser.add_argument('--save_vis', action='store_true')
parser.add_argument('--seg12',default=True, action='store_true')

parser.add_argument('--hf_dataset_identifier', default="florian-morel22/test", type=str)
parser.add_argument(
    "--splitted_celeba_path",
    type=str,
    help="path towards the csv file with the celeba splitting",
    default="dataset/celeba-dataset/versions/2/splitted_celeba.csv"
)
parser.add_argument('--model_name', default="", type=str, help="name of the model name used in the celeba-splitted.csv to pick paires to generate.")
parser.add_argument('--start', default=0, type=int, help="Numerous of data to start from during generation")
parser.add_argument('--stop', default=1000, type=int, help="Numerous of data to stop at during generation")

opt = parser.parse_args()
print(opt)
if opt.laion400m:
    print("Falling back to LAION 400M model...")
    opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    opt.ckpt = "models/ldm/text2img-large/model.ckpt"
    opt.outdir = "outputs/txt2img-samples-laion400m"

seed_everything(opt.seed)

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = model.to(device)

if opt.plms:
    sampler = PLMSSampler(model)
else:
    sampler = DDIMSampler(model)


os.makedirs(opt.outdir, exist_ok=True)
outpath = opt.outdir
data_folder = opt.data_folder
splitted_celeba_path = opt.splitted_celeba_path
hf_dataset_identifier = opt.hf_dataset_identifier
model_name = opt.model_name if opt.model_name != "" else None

batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = list(chunk(data, batch_size))

# sample_path = os.path.join(outpath, "samples")
result_path = os.path.join(outpath, "images")
log_path = os.path.join(outpath)

os.makedirs(result_path, exist_ok=True)

faceParsing_model = init_faceParsing_pretrained_model(opt.faceParser_name, opt.faceParsing_ckpt, opt.segnext_config)

def run_inference(scale=3.5, steps=50):
    opt.ddim_steps=steps
    opt.scale=scale

    conf_file=OmegaConf.load(opt.config)   

    test_args=conf_file.data.params.test.params        
    
    test_dataset=OurCelebADataset(
        faceParsing_model=faceParsing_model,
        faceParser_name=opt.faceParser_name,
        seg12=opt.seg12,
        data_path=data_folder,
        splitted_celeba_path=splitted_celeba_path,
        model_name=model_name,
        start=opt.start,
        stop=opt.stop,
        **test_args
    )
    test_dataset.process_data()

    test_dataloader= torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=True, 
        shuffle=False, 
        drop_last=False
    )


    start_code = None
    if opt.fixed_code:
        print("Using fixed code.......")
        start_code = torch.randn([ opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        # extend the start code to batch size
        start_code = start_code.unsqueeze(0).repeat(batch_size, 1, 1, 1)


    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    saved_results = []

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    sample=0
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():

                for data_ in test_dataloader:

                    sample+=opt.n_samples

                    path_orig_target = data_[5]
                    if path_orig_target[0] == "ERROR":
                        continue

                    target_img = data_[0]
                    source_img = data_[1]
                    test_model_kwargs = data_[2]
                    segment_id_batch = data_[3]
                    inv_transforms = data_[4]
                    target_id = data_[6]
                    source_id = data_[7]

                    test_model_kwargs={n:test_model_kwargs[n].to(device,non_blocking=True) for n in test_model_kwargs }
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector.repeat(target_img.shape[0],1,1)
                        if model.stack_feat:
                            uc2=model.other_learnable_vector.repeat(target_img.shape[0],1,1)
                            uc=torch.cat([uc,uc2],dim=-1)
                    
                    # c = model.get_learned_conditioning(test_model_kwargs['ref_imgs'].squeeze(1).to(torch.float16))
                    landmarks=model.get_landmarks(target_img) if model.Landmark_cond else None
                    ref_imgs=source_img.to(device, non_blocking=True)
                    
                    c=model.conditioning_with_feat(ref_imgs.squeeze(1).to(torch.float32),landmarks=landmarks,tar=target_img.to(device).to(torch.float32)).float()
                    if (model.land_mark_id_seperate_layers or model.sep_head_att) and opt.scale != 1.0:
            
                        # concat c, landmarks
                        landmarks=landmarks.unsqueeze(1) if len(landmarks.shape)!=3 else landmarks
                        uc=torch.cat([uc,landmarks],dim=-1)
                    
                    
                    if c.shape[-1]==1024:
                        c = model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)
                    inpaint_image=test_model_kwargs['inpaint_image']
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    # breakpoint()
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=target_img.shape[0],
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs,src_im=ref_imgs.squeeze(1).to(torch.float32),tar=target_img.to(device))
                        
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
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


                    for i, x_sample in enumerate(x_checked_image_torch):
                        

                        all_img=[]
                        all_img.append(un_norm(target_img[i]).cpu())
                        all_img.append(un_norm(inpaint_image[i]).cpu())
                        ref_img=ref_imgs.squeeze(1)
                        ref_img=Resize([512,512])(ref_img)
                        all_img.append(un_norm_clip(ref_img[i]).cpu())
                        all_img.append(x_sample)
                        grid = torch.stack(all_img, 0)
                        grid = make_grid(grid)
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))

                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8)).resize((1024,1024), Image.BILINEAR)
                        
                        orig_image=Image.open(path_orig_target[0])
                        inv_trans_coeffs = inv_transforms[0][0].tolist()
                        swapped_and_pasted = img.convert('RGBA')
                        pasted_image = orig_image.convert('RGBA')
                        swapped_and_pasted.putalpha(255)

                        projected = swapped_and_pasted.transform(orig_image.size, Image.PERSPECTIVE, inv_trans_coeffs, Image.BILINEAR)
                        pasted_image.alpha_composite(projected)
                        #remove alpha channel
                        pasted_image = pasted_image.convert('RGB')
                    
                        # save pasted image
                        id_pasted_image = f"{source_id[i]}_{target_id[i]}_Reface"
                        pasted_image.save(os.path.join(result_path, id_pasted_image + ".jpg"))

                        saved_results.append({
                            "id": id_pasted_image,
                            "image": pasted_image,
                            "fake": 1,
                            "model": "REFace",
                            "id_source": source_id[i],
                            "id_target": target_id[i],
                        })

    update_dataset(saved_results, hf_dataset_identifier)


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    run_inference()
