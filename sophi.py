'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.videoswap import video_swap
import os
from pathlib import Path
from time import sleep
from random import shuffle

def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0

transformer = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# detransformer = transforms.Compose([
#         transforms.Normalize([0, 0, 0], [1/0.229, 1/0.224, 1/0.225]),
#         transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
#     ])

def spicoutvid(inimgpath,vpath):
    opt = TestOptions().parse()

    start_epoch, epoch_iter = 1, 0
    crop_size = opt.crop_size

    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    model = create_model(opt)
    model.eval()

    
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode=mode)

    with torch.no_grad():
        pic_a = inimgpath
        outpath = Path(r'D:\Developed\VFS\RandyVideo\xdivision')
        opt.output_path = str(outpath / (Path(pic_a).stem + Path(vpath).name))
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _ = app.get(img_a_whole,crop_size)
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # pic_b = opt.pic_b_path
        # img_b_whole = cv2.imread(pic_b)
        # img_b_align_crop, b_mat = app.get(img_b_whole,crop_size)
        # img_b_align_crop_pil = Image.fromarray(cv2.cvtColor(img_b_align_crop,cv2.COLOR_BGR2RGB)) 
        # img_b = transformer(img_b_align_crop_pil)
        # img_att = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        video_swap(vpath, latend_id, model, app, opt.output_path,temp_results_dir=opt.temp_path,\
            no_simswaplogo=opt.no_simswaplogo,use_mask=opt.use_mask,crop_size=crop_size)

if __name__ == '__main__':
    srcimgdir = Path(r'D:\paradise\stuff\simswappg\srcs')
    # dstvideodir = Path(r'D:\paradise\stuff\simswappg\targets')
    # dstvideodir = Path(r'D:\paradise\stuff\new\PVD\extractedVideo')
    # dstvideodir = Path(r'D:\paradise\stuff\new\pvd2\Ginnibhabhi_video')
    dstvideodir = Path(r'D:\paradise\stuff\new\PVD\Yummyx (17)_video')

    # targetfile = open('donedata.csv','w+')
    testsrc_times = -1
    randsrc = True
    randdst = True
    # targetfile = open('donedata.csv','w+')
    srcFileList = [x for x in srcimgdir.glob('*.jpg')]
    dstFileList = [x for x in dstvideodir.glob('*.mp4')]
    if randsrc:
        shuffle(srcFileList)    
        
    for imgFiles in srcFileList:
      parentdir = imgFiles.parent / 'VFsRecords'
      parentdir.mkdir(exist_ok=True)  
      dbfilename = parentdir / (imgFiles.stem+'.csv')
      donedata = open(dbfilename, 'a+') 
      donedata.seek(0,0)  
      fcontent = [x.strip() for x in donedata.readlines()]
      # import pdb;pdb.set_trace()
      donedata.close()
      setfcontent = set(fcontent)
      tsc = testsrc_times
      if randdst:
          shuffle(dstFileList)
      for vidFIle in dstFileList:
          if tsc == 0:
            break
          if str(vidFIle) not in setfcontent:
            try:
                spicoutvid(str(imgFiles), str(vidFIle))
                donedata = open(dbfilename, 'a+')
                donedata.write('\n'+ str(vidFIle)) 
                donedata.close()
            except:
                print(str(imgFiles), str(vidFIle))
            tsc -= 1
            # sleep(1000)
          else:
            print('already done')
            continue       
