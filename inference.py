import glob
import argparse
import os

import random
import cv2
import insightface
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import torch
from models_sw import *
from utils import *

from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
assert insightface.__version__ >= '0.3'

parser = argparse.ArgumentParser()

parser.add_argument('--ctx', default=0, type=int,
                    help='ctx id, <0 means using cpu')
parser.add_argument('--det-size', default=640, type=int, help='detection size')

parser.add_argument('--gpu', type=str, default='0', help='gpu id')
parser.add_argument('--input', type=str, default='./test_imgs/trump.jpg',
                    help='Input image path. Require JPG or PNG file.')
parser.add_argument('--model_path', type=str, default='./checkpoints/',
                    help='Path to save checkpoints.')
# BEST_checkpointcuda:1.pth GWT.pth
parser.add_argument('--ckpt', type=str, default="./checkpoints/GWT.pth")

parser.add_argument('--num_class', type=int, default=26)
parser.add_argument('--num_block1', type=int, default=1)
parser.add_argument('--num_block2', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.00006)
parser.add_argument('--inside_lr', type=float, default=1)
parser.add_argument('--step', type=float, default=3)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.96)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--grad_clip', type=float, default=None)
parser.add_argument('--display_freq', type=int, default=20)
parser.add_argument('--eval_freq', type=int, default=20)
parser.add_argument('--loss_type', type=str, default='bce',
                    choices=['l2', 'multilabel', 'bce', 'focal', 'balance_focal'])
parser.add_argument('--schedule', type=str, default='nocycle')
parser.add_argument('--innerLoss', type=str, default='BCE')
parser.add_argument('--grad_d_weight', type=float, default='0.1')

args = parser.parse_args()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

app = FaceAnalysis(name='buffalo_l', root='./thirdparty/insightface',
                   providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=int(args.gpu), det_size=(args.det_size, args.det_size))

detector = insightface.model_zoo.get_model(
    'scrfd_person_2.5g.onnx', download=True)
detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))

emotic_CLASSES = ['affection', 'anger', 'annoyance', 'anticipation', 'aversion', 'confidence', 'disapproval', 'disconnection',
                  'disquietment', 'doubt / confusion', 'embarrassment', 'engagement', 'esteem', 'excitement', 'fatigue', 'fear',
                  'happiness', 'pain', 'peace', 'pleasure', 'sadness', 'sensitivity', 'suffering', 'surprise', 'sympathy', 'yearning']


def validate_bbox(im_size, bbox):
    x1, y1, x2, y2 = bbox
    x1 = min(im_size[1], max(0, x1))
    x2 = min(im_size[1], max(0, x2))
    y1 = min(im_size[0], max(0, y1))
    y2 = min(im_size[0], max(0, y2))
    return [int(x1), int(y1), int(x2), int(y2)]


def scale_coord(coord, src, tgt):
    x1, y1, x2, y2 = coord
    h0, w0 = src
    h1, w1 = tgt

    x1 *= w1 / w0
    y1 *= h1 / h0
    x2 *= w1 / w0
    y2 *= h1 / h0

    return (int(x1), int(y1), int(x2), int(y2))


def detect_person(img, detector):
    bboxes, kpss = detector.detect(img)
    bboxes = np.round(bboxes[:, :4]).astype(np.int)
    kpss = np.round(kpss).astype(np.int)
    kpss[:, :, 0] = np.clip(kpss[:, :, 0], 0, img.shape[1])
    kpss[:, :, 1] = np.clip(kpss[:, :, 1], 0, img.shape[0])
    vbboxes = bboxes.copy()
    vbboxes[:, 0] = kpss[:, 0, 0]
    vbboxes[:, 1] = kpss[:, 0, 1]
    vbboxes[:, 2] = kpss[:, 4, 0]
    vbboxes[:, 3] = kpss[:, 4, 1]
    return bboxes, vbboxes


def draw_emotions(img, face_rect, emotions=None):

    # img = image.copy()

    x, y, w, h = face_rect
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (0, 255, 0)  
    bg_color = (0, 0, 0)      
    line_spacing = 10         
    
   
    start_x = x + w + 10  
    start_y = y           
    

    for i, emotion in enumerate(emotions):
        
        current_y = start_y + i * line_spacing

        text_size = cv2.getTextSize(emotion, font, font_scale, font_thickness)[0]

        cv2.rectangle(
            img, 
            (start_x - 2, current_y - text_size[1] - 2),
            (start_x + text_size[0] + 2, current_y + 2),
            bg_color, 
            -1  
        )
    
        cv2.putText(
            img, 
            emotion, 
            (start_x, current_y),
            font, 
            font_scale, 
            text_color, 
            font_thickness
        )


def draw_rectangle_with_number(image, x1, y1, number, color=(0, 255, 0), thickness=2, font_scale=0.8):

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(number)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x1
    text_y = y1 + 10
    cv2.putText(image, text, (text_x, text_y),
                font, font_scale, color, thickness)

    return image


def emotion_CocoER_output(img, coord_list):

    output_dict_final = {}

    for k, it in enumerate(coord_list):
        x1, y1, x2, y2, x11, y11, x21, y21 = it[0], it[1], it[2], it[3], it[4], it[5], it[6], it[7]
        print(x1, y1, x2, y2, x11, y11, x21, y21)
        img_ = copy.deepcopy(img)
        raw_context = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4)])(img_)
        raw_body = copy.deepcopy(raw_context.crop((x1, y1, x2, y2)))
        raw_head = copy.deepcopy(raw_context.crop((x11, y11, x21, y21)))

        coord_body = [x1, y1, x2, y2]
        coord_head = [x11, y11, x21, y21]

        ctx_img = trans_base(raw_context)
        body_img = trans_base(raw_body)
        head_img = trans_base(raw_head)

        body_coord = scale_coord(coord_body, (H, W), (224, 224))
        head_coord = scale_coord(coord_head, (H, W), (224, 224))

        images_context = ctx_img.unsqueeze(0).to(device)
        images_body = body_img.unsqueeze(0).to(device)
        images_head = head_img.unsqueeze(0).to(device)
        coord_body = torch.tensor(body_coord).unsqueeze(0).to(device)
        coord_head = torch.tensor(head_coord).unsqueeze(0).to(device)

        pred, _, ret = model(images_context, images_body,
                             images_head, coord_body, coord_head, None)
        pred = nn.Sigmoid()(pred)
        a_ = (torch.ones(pred[0].shape) * 0.6).to(device)
        label_VI = nn.Sigmoid()(ret['label_VI'])
        label_head = nn.Sigmoid()(ret['label_head'])
        label_body = nn.Sigmoid()(ret['label_body'])
        label_ctx = nn.Sigmoid()(ret['label_ctx'])

        # label_head_refined = ret['label_head_refined']
        # label_body_refined = ret['label_body_refined']
        # label_ctx_refined = ret['label_ctx_refined']

        pred_VI = torch.gt(label_VI, a_).type(torch.float32)
        pred_head = torch.gt(label_head, a_).type(torch.float32)
        pred_body = torch.gt(label_body, a_).type(torch.float32)
        pred_ctx = torch.gt(label_ctx, a_).type(torch.float32)

        # pred_head_refined = torch.gt(label_head_refined, a_).type(torch.float32)
        # pred_body_refined = torch.gt(label_body_refined, a_).type(torch.float32)
        # pred_ctx_refined = torch.gt(label_ctx_refined, a_).type(torch.float32)

        pred_final = torch.gt(
            pred, (torch.ones(pred[0].shape) * 0.26).to(device)).type(torch.float32)
        output_dict = {}
        pred_VI_li, pred_head_li, pred_body_li, pred_ctx_li, pred_final_li, = [], [], [], [], []
        for ii, item in enumerate(emotic_CLASSES):
            if pred_final[0][ii] == 1.:
                pred_final_li.append(item)
            if pred_VI[0][ii] == 1.:
                pred_VI_li.append(item)
            if pred_head[0][ii] == 1.:
                pred_head_li.append(item)
            if pred_body[0][ii] == 1.:
                pred_body_li.append(item)
            if pred_ctx[0][ii] == 1.:
                pred_ctx_li.append(item)

        output_dict['pred'] = pred_final_li
        output_dict['VI'] = pred_VI_li
        output_dict['head'] = pred_head_li
        output_dict['body'] = pred_body_li
        output_dict['ctx'] = pred_ctx_li
        output_dict['emo_process'] = ret['ret_emo_process_all'][0]

        print(f"\n\n*** person ID={str(k)} ***")
        print(f"head-level prediction -> {pred_head_li}")
        print(f"body-level prediction -> {pred_body_li}")
        print(f"context-level prediction -> {pred_ctx_li}")
        print(f"vocabulary-informed prediction -> {pred_VI_li}")
        print(f"=== final prediction ===  {pred_final_li}")
        print(f"=== exclusion sequence === {output_dict['emo_process']} \n\n")

        output_dict_final[str(k)] = output_dict

    return output_dict_final


if __name__ == "__main__":

    img_path = args.input
    img = cv2.imread(img_path)
    
    try:
        bboxes, vbboxes = detect_person(img, detector)
        img_face = ins_get_image(os.path.abspath(img_path)[:-4])
        faces = app.get(img_face)
    except:
        print(" Please check photo dir. ")
        exit()

    if len(faces) == 0:
        print(" No face detected. ")
        exit()

    coord_list = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        bbox = validate_bbox(img.shape, bbox)
        x1, y1, x2, y2 = bbox
        status = 0
        for item in faces:

            x11, y11, x21, y21 = validate_bbox(img.shape, item.bbox)
            print(x1, y1, x2, y2)
            mid_x = (x11 + x21) // 2
            if mid_x >= x1 and mid_x <= x2 and x11 >= x1 and x21 <= x2:
                status = 1
                break
        if status == 1:
            coord_list.append(
                [x1, y1, x2, y2, x11, y11, x21, y21])  # body, face

    if len(coord_list) == 0:
        print(" No person dectected. Please check the photo. ")
        exit()

    # load model
    device = f"cuda:{args.gpu}"
    model = model_GWT(args).to(device)
    ckpt = args.ckpt
    ckpt_state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_state_dict['model'])

    trans_base = transforms.Compose([
                                    transforms.Resize(
                                        size=(224, 224)),  # 224,224
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                    ])



    H, W, _ = img.shape

    output = emotion_CocoER_output(img, coord_list)
    img_draw = copy.deepcopy(img)
    for i, it in enumerate(coord_list):
        cv2.rectangle(img_draw, (it[0], it[1]), (it[2], it[3]), (0, 255, 0), 1)
        cv2.rectangle(img_draw, (it[4], it[5]),
                      (it[6], it[7]), (0, 255, 255), 1)
        draw_rectangle_with_number(img_draw, it[0], it[1], i, (255, 0, 0))
        draw_emotions(img_draw, (it[4],it[5],it[6]-it[4],it[7]-it[5]), emotions=output[str(i)]["pred"])

    cv2.imwrite(f'./outputs/{os.path.basename(img_path)}', img_draw)
    
