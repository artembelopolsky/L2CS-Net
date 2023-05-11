# %%

import argparse
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS

import os
import csv
# %%

# To run: python demo_jpg.py --snapshot models/L2CSNet_gaze360.pkl --gpu 0
# %%
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    
    try:
        args = parse_args()
        cudnn.enabled = True
        arch=args.arch
        batch_size = 1
        cam = args.cam_id
        gpu = select_device(args.gpu_id, batch_size=batch_size)
        snapshot_path = args.snapshot
   
    except:        
        cudnn.enabled = True
        arch='ResNet50'
        batch_size = 1
        cam = 0
        gpu = select_device(0, batch_size=batch_size)
        snapshot_path = 'models/L2CSNet_gaze360.pkl'
   
    

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path,map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
    #model.cuda()
    model.eval()


    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=-1)
    idx_tensor = [idx for idx in range(90)]
    # idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    idx_tensor = torch.FloatTensor(idx_tensor)
    x=0
  
    # cap = cv2.VideoCapture(cam)

    # # Check if the webcam is opened correctly
    # if not cap.isOpened():
    #     raise IOError("Cannot open webcam")

    
    fnames = os.listdir('./movies/frames/sample/')

    print(fnames)

    # open output file for writing   
    with open('./movies/frames/output_csv/out_' + fnames[0].split('.')[0] + '.csv', 'w', newline='') as outf:
        # create the csv writer
        writer = csv.writer(outf)
        header = ['frame', 'yaw', 'pitch', 'bbox_area']
        writer.writerow(header)
        # get gaze    
        with torch.no_grad():
            for f in fnames:           
            
                # used to sort face bboxes
                bbox_param = []

                print(f'Reading file: {f}...')
                frame = cv2.imread('./movies/frames/sample/' + f)
                faces = detector(frame)
                if faces is not None: 
                    for box, landmarks, score in faces:
                        if score < .95:
                            continue
                        x_min=int(box[0])
                        if x_min < 0:
                            x_min = 0
                        y_min=int(box[1])
                        if y_min < 0:
                            y_min = 0
                        x_max=int(box[2])
                        y_max=int(box[3])
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        bbox_area = bbox_width * bbox_height
                        # x_min = max(0,x_min-int(0.2*bbox_height))
                        # y_min = max(0,y_min-int(0.2*bbox_width))
                        # x_max = x_max+int(0.2*bbox_height)
                        # y_max = y_max+int(0.2*bbox_width)
                        # bbox_width = x_max - x_min
                        # bbox_height = y_max - y_min

                        # collect all bbox parameters for future sorting
                        bbox_param.append([bbox_area, x_min, x_max, y_min, y_max])

                    # get three largest bounding boxes
                    bbox_param  = np.array(bbox_param)
                    bbox_param = bbox_param[bbox_param[:, 0].argsort()][::-1] # sort based bbox_area in descending order
                    bbox_param = bbox_param[:3] # select subset largest bboxes

                    for bbox_area, x_min, x_max, y_min, y_max in bbox_param:


                        # Crop image
                        img = frame[y_min:y_max, x_min:x_max]
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(img)
                        img=transformations(im_pil)
                        # img  = Variable(img).cuda(gpu)
                        img  = Variable(img)
                        img  = img.unsqueeze(0) 
                        
                        # gaze prediction
                        gaze_pitch, gaze_yaw = model(img)
                        
                        
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)

                                        
                        # Get continuous predictions in degrees.
                        pitch_predicted_deg = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                        yaw_predicted_deg = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                        # convert to numpy
                        pitch_predicted_deg = pitch_predicted_deg.cpu().detach().numpy()
                        yaw_predicted_deg = yaw_predicted_deg.cpu().detach().numpy()

                        row = [f, yaw_predicted_deg, pitch_predicted_deg, bbox_area]
                        print(f'Yaw predicted in degrees: {yaw_predicted_deg}, Pitch predicted in degrees: {pitch_predicted_deg}, Bbox_area: {bbox_area}')
                        writer.writerow(row)
                                            
                        pitch_predicted= pitch_predicted_deg * np.pi/180.0
                        yaw_predicted= yaw_predicted_deg * np.pi/180.0
                        
                        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch_predicted,yaw_predicted),color=(0,0,255))
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                        
                
                
                cv2.imwrite('./movies/frames/output/' + 'out_' + f, frame)
        
            

# %%