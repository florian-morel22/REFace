# Author: Sanoojan

import os
import sys
import cv2
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm

print("\n>> Process CelebA mask\n")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('folder_anno_masks', type=str, default='CelebAMask-HQ', help='folder with mask annotations')
parser.add_argument('save_path', type=str, default='CelebAMask-HQ', help='folder to save masks')
args = parser.parse_args()

folder_anno_masks = args.folder_anno_masks
save_path = args.save_path


# 19 attributes in total, skin-1,nose-2,...cloth-18, background-0
celelbAHQ_label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                        'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r',
                        'neck_l', 'neck', 'cloth']

# 12 attributes with left-right aggrigation
faceParser_label_list_detailed = ['background', 'lip', 'eyebrows', 'eyes', 'hair', 
                                  'nose', 'skin', 'ears', 'belowface', 'mouth', 
                                  'eye_glass', 'ear_rings']


if not os.path.exists(save_path):
    os.mkdir(save_path)

assert os.path.exists(folder_anno_masks), "Path does not exist"

for i in tqdm(range(30000)):
    #  create blank image with 512,512

    mask=np.zeros((512,512))
    for ind, cate in enumerate(celelbAHQ_label_list):
        # check if path exists s.path.join(Dataset_maskPath,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')
        

        if os.path.exists(os.path.join(folder_anno_masks,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')):
            im= Image.open(os.path.join(folder_anno_masks,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')).convert('L')
            im = np.equal(im, 255)
            mask[im]=ind+1
        # else:
        #     # print("image not exists")
        #     # print(os.curdir)
        #     print(os.path.exists(os.path.join(folder_anno_masks,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png')))
        #     # print(os.path.join(Dataset_maskPath,"%d"%int(i/2000) ,'{0:0=5d}'.format(i)+'_'+cate+'.png'))
    # save the mask
    cv2.imwrite(os.path.join(save_path,'%d'%i+'.png'),mask)


