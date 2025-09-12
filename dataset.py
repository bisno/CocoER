import os
import random
import cv2
import copy
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import pickle
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import warnings
import scipy.io as scio
import matplotlib.pyplot as plt

from skimage import img_as_ubyte, img_as_float


class Emotic_PreDataset(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, args, mode='train', size_type='256'):
        super(Emotic_PreDataset, self).__init__()
        # Load preprocessed data from npy files
        self.x_context = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask{}/{}_context_arr{}.npy'.format(size_type, mode, size_type)))
        self.x_body = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask{}/{}_body_arr{}.npy'.format(size_type, mode, size_type)))
        self.x_head = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask{}/{}_head_arr{}.npy'.format(size_type, mode, size_type)))
        self.y_cat = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask{}/{}_cat_arr{}.npy'.format(size_type, mode, size_type)))
        # label_smtc_file = os.path.join(args.data_path, 'label_semantic_embeddings.pkl')
        # self.label_emb = pickle.load(open(label_smtc_file, 'rb'))

        self.mode = mode
        if 'train' in mode:
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [transforms.ToPILImage(), transforms.ToTensor()])

        self.norm_trans = transforms.Normalize(
            IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        # self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])  # Normalizing the context image with context mean and context std
        # self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])  # Normalizing the body image with body mean and body std
        # self.head_norm = transforms.Normalize(head_norm[0], head_norm[1])

    def get_label_emb(self):
        labels_cls_embs = []
        ind2emb = self.label_emb['ind2emb']
        for key in ind2emb:
            emb = ind2emb[key]['word_emb']
            labels_cls_embs.append(emb.detach().numpy())
        labels_cls_embs = torch.tensor(
            np.array(labels_cls_embs), dtype=torch.float32)
        return labels_cls_embs

    def get_transform(self, flip_p=False, crop=224, RE=[0.0, 0.0, 0.0], colorjitter=None):
        if self.mode == 'train':
            transform_List = [transforms.ToPILImage()]
            if crop is not None:
                transform_List.append(transforms.RandomCrop(size=crop))
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            if colorjitter is not None:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__jitter__(img, colorjitter)))
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            if RE is not None:
                transform_List.append(
                    transforms.RandomErasing(p=0.4, value=RE))
            self.transform = transforms.Compose(transform_List)
        else:
            transform_List = [transforms.ToPILImage(
            ), transforms.Resize(size=224, interpolation=3)]
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            self.transform = transforms.Compose(transform_List)

        return self.transform

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def __jitter__(img, params):
        brightness, contrast, saturation, hue = params
        if isinstance(img, np.ndarray):

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage(
            )] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = img
                for func in img_transforms:
                    jittered_img = func(jittered_img)
                jittered_img = jittered_img.astype('float32')

        elif isinstance(img, Image.Image):
            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img)))
        return jittered_img

    @staticmethod
    def get_jitter_params(brightness=0, contrast=0, saturation=0, hue=0):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        image_head = self.x_head[index]
        cat_label = self.y_cat[index]

        flip_p = random.random() < 0.5
        jitter_p = self.get_jitter_params(
            brightness=0.4, contrast=0.4, saturation=0.4)
        trans = self.get_transform(flip_p, crop=224, RE=[
                                   0.0, 0.0, 0.0], colorjitter=jitter_p)

        # ctx_info = self.norm_trans(self.transform(image_context))
        # bd_info = self.norm_trans(self.transform(image_body))
        # head_info = self.norm_trans(self.transform(image_head))
        ctx_info = trans(image_context)
        bd_info = trans(image_body)
        head_info = trans(image_head)

        cat_info = torch.tensor(cat_label, dtype=torch.float32)
        return ctx_info, bd_info, head_info, cat_info


class Emotic_PreDataset2(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, args, mode='train', size_type='224'):
        super(Emotic_PreDataset2, self).__init__()
        # Load preprocessed data from npy files
        self.x_context = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask_coord{}/{}_context_arr{}.npy'.format(size_type, mode, size_type)))
        self.x_body = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask_coord{}/{}_body_arr{}.npy'.format(size_type, mode, size_type)), allow_pickle=True)
        self.x_head = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask_coord{}/{}_head_arr{}.npy'.format(size_type, mode, size_type)), allow_pickle=True)
        self.y_cat = np.load(os.path.join(
            args.data_path, 'emotic_pre_mask_coord{}/{}_cat_arr{}.npy'.format(size_type, mode, size_type)))
        # label_smtc_file = os.path.join(args.data_path, 'label_semantic_embeddings.pkl')
        # self.label_emb = pickle.load(open(label_smtc_file, 'rb'))

        self.mode = mode

    def get_label_emb(self):
        labels_cls_embs = []
        ind2emb = self.label_emb['ind2emb']
        for key in ind2emb:
            emb = ind2emb[key]['word_emb']
            labels_cls_embs.append(emb.detach().numpy())
        labels_cls_embs = torch.tensor(
            np.array(labels_cls_embs), dtype=torch.float32)
        return labels_cls_embs

    def get_transform(self, flip_p=False, crop=224, RE=[0.0, 0.0, 0.0], colorjitter=None):
        if self.mode == 'train':
            transform_List = [transforms.ToPILImage()]
            if crop is not None:
                transform_List.append(transforms.RandomCrop(size=crop))
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            if colorjitter is not None:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__jitter__(img, colorjitter)))
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            if RE is not None:
                transform_List.append(
                    transforms.RandomErasing(p=0.4, value=RE))
            self.transform = transforms.Compose(transform_List)
        else:
            transform_List = [transforms.ToPILImage()]
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            self.transform = transforms.Compose(transform_List)

        return self.transform

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def __jitter__(img, params):
        brightness, contrast, saturation, hue = params
        if isinstance(img, np.ndarray):

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage(
            )] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = img
                for func in img_transforms:
                    jittered_img = func(jittered_img)
                jittered_img = jittered_img.astype('float32')

        elif isinstance(img, Image.Image):
            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img)))
        return jittered_img

    @staticmethod
    def get_jitter_params(brightness=0, contrast=0, saturation=0, hue=0):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __len__(self):
        return len(self.y_cat)

    def __getitem__(self, index):
        image_context = self.x_context[index]
        image_body, body_coord = self.x_body[index]  # [], [x1y1x2y2]
        image_head, head_coord = self.x_head[index]  # [], [x1y1x2y2]
        cat_label = self.y_cat[index]

        H, W, _ = image_context.shape
        body_coord[0] = max(body_coord[0], 0)
        body_coord[1] = max(body_coord[1], 0)
        body_coord[2] = min(body_coord[2], W)
        body_coord[3] = min(body_coord[3], H)
        head_coord[0] = max(head_coord[0], 0)
        head_coord[1] = max(head_coord[1], 0)
        head_coord[2] = min(head_coord[2], W)
        head_coord[3] = min(head_coord[3], H)

        flip_p = random.random() < 0.5
        jitter_p = self.get_jitter_params(
            brightness=0.4, contrast=0.4, saturation=0.4)
        trans = self.get_transform(
            flip_p, crop=None, RE=None, colorjitter=jitter_p)

        # ctx_info = self.norm_trans(self.transform(image_context))
        # bd_info = self.norm_trans(self.transform(image_body))
        # head_info = self.norm_trans(self.transform(image_head))
        ctx_info = trans(image_context)
        bd_info = trans(image_body)
        head_info = trans(image_head)

        cat_info = torch.tensor(cat_label, dtype=torch.float32)
        return ctx_info, bd_info, head_info, cat_info, torch.tensor(body_coord), torch.tensor(head_coord)


class Emotic_PreDataset3(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, data_path, mode='train', size_type='224', use_all=True):
        super(Emotic_PreDataset3, self).__init__()
        # Load preprocessed data from npy files

        self.info = np.load(os.path.join(
            data_path, 'emotic_pre_coord{}/{}_info.npy'.format(size_type, mode)), allow_pickle=True).item()
        self.mode = mode
        self.use_all = use_all

        if mode in ['val', 'test']:
            assert use_all
        self.precoess()

    def precoess(self):
        data = []

        for k in self.info.keys():
            seqs = self.info[k]

            L = len(seqs)
            if L > 1:
                indices = np.random.choice(
                    range(L), size=L if self.use_all else 1, replace=False)
                seq = [seqs[idx] for idx in indices]
            else:
                seq = seqs
            data += seq
        random.shuffle(data)
        self.data = data

    def get_transform(self, flip_p=False, crop=224, RE=[0.0, 0.0, 0.0], colorjitter=None):
        if self.mode in ['train', 'trainval']:
            transform_List = [transforms.ToPILImage()]
            if crop is not None:
                transform_List.append(transforms.RandomCrop(size=crop))
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            if colorjitter is not None:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__jitter__(img, colorjitter)))
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            if RE is not None:
                transform_List.append(
                    transforms.RandomErasing(p=0.4, value=RE))
            self.transform = transforms.Compose(transform_List)
        else:
            transform_List = [transforms.ToPILImage()]
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            self.transform = transforms.Compose(transform_List)

        return self.transform

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def __jitter__(img, params):
        brightness, contrast, saturation, hue = params
        if isinstance(img, np.ndarray):

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage(
            )] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = img
                for func in img_transforms:
                    jittered_img = func(jittered_img)
                jittered_img = jittered_img.astype('float32')

        elif isinstance(img, Image.Image):
            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img)))
        return jittered_img

    @staticmethod
    def get_jitter_params(brightness=0, contrast=0, saturation=0, hue=0):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    @staticmethod
    def adjust_bbox(im_size, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(im_size[1], max(0, x1))
        x2 = min(im_size[1], max(0, x2))
        y1 = min(im_size[0], max(0, y1))
        y2 = min(im_size[0], max(0, y2))

        return [int(x1), int(y1), int(x2), int(y2)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_context, image_body, image_head, cat_label, _, body_coord, head_coord = self.data[
            index]
        # image_context = cv2.cvtColor(image_context, cv2.COLOR_RGB2BGR)

        H, W, _ = image_context.shape
        body_coord = self.adjust_bbox([H, W], body_coord)
        head_coord = self.adjust_bbox([H, W], head_coord)

        flip_p = random.random() < 0.5
        jitter_p = self.get_jitter_params(
            brightness=0.4, contrast=0.4, saturation=0.4)
        trans = self.get_transform(
            flip_p, crop=None, RE=None, colorjitter=jitter_p)

        ctx_info = trans(image_context)
        bd_info = trans(image_body)
        head_info = trans(image_head)

        cat_info = torch.tensor(cat_label, dtype=torch.float32)
        return ctx_info, bd_info, head_info, cat_info, torch.tensor(body_coord), torch.tensor(head_coord)

# randomcrop


class Emotic_PreDataset4(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, data_path, mode='train', img_size=224, use_all=True):
        super(Emotic_PreDataset4, self).__init__()
        # Load preprocessed data from npy files

        with open(os.path.join(data_path, 'emotic_pre/{}_info.pkl'.format(mode)), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(data_path, 'emotic_pre/{}_image.pkl'.format(mode)), 'rb') as f:
            self.image = pickle.load(f)
        # self.info = np.load(os.path.join(data_path, 'emotic_pre_coord/{}_info.npy'.format(mode)), allow_pickle=True).item()
        with open(os.path.join(data_path, 'emotic_pre/test_info.pkl'.format(mode)), 'rb') as f:
            self.data_ = pickle.load(f)
        with open(os.path.join(data_path, 'emotic_pre/test_image.pkl'.format(mode)), 'rb') as f:
            self.image_ = pickle.load(f)

        self.mode = mode
        self.img_size = img_size
        self.use_all = use_all

    def precoess(self):
        data = []

        for k in self.info.keys():
            seqs = self.info[k]

            L = len(seqs)
            if L > 1:
                indices = np.random.choice(
                    range(L), size=L if self.use_all else 1000, replace=False)
                seq = [seqs[idx] for idx in indices]
            else:
                seq = seqs
            data += seq
        random.shuffle(data)
        self.data = data

    def get_transform_simple(self, flip_p=False):
        if self.mode in ['train', 'trainval']:
            transform_List = [transforms.ToPILImage(), ]
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            transform_List += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                               # transforms.RandomErasing(p=0.4, value=[0.0, 0.0, 0.0])
                               ]  # 0.4 0.4 0.4

            self.transform = transforms.Compose(transform_List)
        else:
            flip_p = False
            transform_List = [transforms.ToPILImage()]
            self.transform = transforms.Compose(transform_List)

        return self.transform, flip_p

    def get_transform(self, flip_p=False, crop=224, RE=[0.0, 0.0, 0.0], colorjitter=None):
        if self.mode in ['train', 'trainval']:
            transform_List = [transforms.ToPILImage()]
            if crop is not None:
                transform_List.append(transforms.RandomCrop(size=crop))
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            if colorjitter is not None:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__jitter__(img, colorjitter)))
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            if RE is not None:
                transform_List.append(
                    transforms.RandomErasing(p=0.4, value=RE))
            self.transform = transforms.Compose(transform_List)
        else:
            transform_List = [transforms.ToPILImage()]
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            self.transform = transforms.Compose(transform_List)

        return self.transform

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def __jitter__(img, params):
        brightness, contrast, saturation, hue = params
        if isinstance(img, np.ndarray):

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage(
            )] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = img
                for func in img_transforms:
                    jittered_img = func(jittered_img)
                jittered_img = jittered_img.astype('float32')

        elif isinstance(img, Image.Image):
            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img)))
        return jittered_img

    @staticmethod
    def get_jitter_params(brightness=0, contrast=0, saturation=0, hue=0):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    @staticmethod
    def adjust_bbox(im_size, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(im_size[1], max(0, x1))
        x2 = min(im_size[1], max(0, x2))
        y1 = min(im_size[0], max(0, y1))
        y2 = min(im_size[0], max(0, y2))

        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def coord_origin_translation(coord, x0, y0):
        x1, y1, x2, y2 = coord

        return (x1 - x0, y1 - y0, x2 - x0, y2 - y0)

    @staticmethod
    def scale_coord(coord, src, tgt):
        x1, y1, x2, y2 = coord
        h0, w0 = src
        h1, w1 = tgt

        x1 *= w1 / w0
        y1 *= h1 / h0
        x2 *= w1 / w0
        y2 *= h1 / h0

        return (int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def enlarge_bbox(coord, scale=[0.8, 1.2]):
        x1, y1, x2, y2 = coord
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        r = random.uniform(scale[0], scale[1])
        h = (y2 - y1) / 2 * r
        w = (x2 - x1) / 2 * r

        xx1 = int(cx - w)
        yy1 = int(cy - h)
        xx2 = int(cx + w) if int(cx + w) > xx1 else xx1 + 1
        yy2 = int(cy + h) if int(cy + h) > yy1 else yy1 + 1

        return [xx1, yy1, xx2, yy2]

    def random_crop(self, image, body_coord, head_coord):
        if self.mode in ['train', 'trainval']:
            h, w, _ = image.shape
            x1, y1, x2, y2 = body_coord

            bg_x1 = random.randint(0, x1)
            bg_y1 = random.randint(0, y1)
            bg_x2 = random.randint(x2, w)
            bg_y2 = random.randint(y2, h)

            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

        return image, body_coord, head_coord

    def random_crop2(self, image, body_coord, head_coord):
        if self.mode in ['train', 'trainval']:
            h, w, _ = image.shape
            x01, y01, x02, y02 = head_coord
            x11, y11, x12, y12 = body_coord

            x21 = random.randint(min(x01, x11), x01)
            y21 = random.randint(min(y01, y11), y01)
            x22 = random.randint(x02, max(x02, x12))
            y22 = random.randint(y02, max(y02, y12))

            bg_x1 = random.randint(0, x21)
            bg_y1 = random.randint(0, y21)
            bg_x2 = random.randint(x22, w)
            bg_y2 = random.randint(y22, h)

            body_coord = (x21, y21, x22, y22)
            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

            body_coord = self.adjust_bbox(image.shape[:2], body_coord)
            head_coord = self.adjust_bbox(image.shape[:2], head_coord)

        return image, body_coord, head_coord

    def random_crop3(self, image, body_coord, head_coord):
        if self.mode in ['train', 'trainval']:
            h, w, _ = image.shape

            head_coord = self.adjust_bbox(
                [h, w], self.enlarge_bbox(head_coord, scale=[0.8, 1.2]))
            body_coord = self.adjust_bbox(
                [h, w], self.enlarge_bbox(body_coord, scale=[1.2, 1.2]))

            x01, y01, x02, y02 = head_coord
            x11, y11, x12, y12 = body_coord

            x21 = random.randint(min(x01, x11), x01)
            y21 = random.randint(min(y01, y11), y01)
            x22 = random.randint(x02, max(x02, x12))
            y22 = random.randint(y02, max(y02, y12))

            bg_x1 = random.randint(0, x21)
            bg_y1 = random.randint(0, y21)
            bg_x2 = random.randint(x22, w)
            bg_y2 = random.randint(y22, h)

            body_coord = (x21, y21, x22, y22)
            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

            body_coord = self.adjust_bbox(image.shape[:2], body_coord)
            head_coord = self.adjust_bbox(image.shape[:2], head_coord)

        return image, body_coord, head_coord

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, cat_label, _, body_coord, head_coord = self.data[index]
        image = self.image[key]

        image, body_coord, head_coord = self.random_crop2(
            image, body_coord, head_coord)
        H, W, _ = image.shape

        flip_p = random.random() < 0.5
        trans, flip_p = self.get_transform_simple(flip_p)
        ctx_info = trans(image)
        if flip_p:
            body_coord = (W - body_coord[2], body_coord[1],
                          W - body_coord[0], body_coord[3])
            head_coord = (W - head_coord[2], head_coord[1],
                          W - head_coord[0], head_coord[3])
        body_info = copy.deepcopy(ctx_info.crop(body_coord))
        head_info = copy.deepcopy(ctx_info.crop(head_coord))
        cat_info = torch.tensor(cat_label, dtype=torch.float32)

        # totensor + normalize
        # trans_base = transforms.Compose([transforms.Resize(size=(self.img_size, self.img_size)),
        #                                  transforms.ToTensor(),
        #                                  transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        #                                  # transforms.RandomErasing(p=0.4, value=[0.0, 0.0, 0.0])
        #                                  ])

        trans_base = transforms.Compose([transforms.Resize(size=(224, 224)),  # 224,224
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                         # transforms.RandomErasing(p=0.4, value=[0.0, 0.0, 0.0])
                                         ])  # 2024 -> resize

        ctx_img = trans_base(ctx_info)
        body_img = trans_base(body_info)
        head_img = trans_base(head_info)

        body_coord = self.scale_coord(
            body_coord, (H, W), (self.img_size, self.img_size))
        head_coord = self.scale_coord(
            head_coord, (H, W), (self.img_size, self.img_size))

        # --- debug ---
        # cv2.imwrite('./emotion/train_transimg/ctx_img.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey() image.permute(1,2,0).numpy() cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('./emotion/train_transimg/body_img.jpg', cv2.cvtColor(image[body_coord[1]:body_coord[3],body_coord[0]:body_coord[2]], cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        # cv2.imwrite('./emotion/train_transimg/head_img.jpg', cv2.cvtColor(head_info, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()

        return ctx_img, body_img, head_img, cat_info, torch.tensor(body_coord), torch.tensor(head_coord)


class Emotic_PreDataset4_CAER(Dataset):
    ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''

    def __init__(self, data_path, mode='train', img_size=224, use_all=True):
        super(Emotic_PreDataset4_CAER, self).__init__()
        # Load preprocessed data from npy files
        mode = 'train'
        with open(os.path.join(data_path, '{}_info.pkl'.format(mode)), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(data_path, '{}_image.pkl'.format(mode)), 'rb') as f:
            self.image = pickle.load(f)
        # self.info = np.load(os.path.join(data_path, 'emotic_pre_coord/{}_info.npy'.format(mode)), allow_pickle=True).item()
        with open('./datasets/CAER-S/CAER-S-processed/test_info.pkl', 'rb') as f:
            self.data_ = pickle.load(f)
        with open('./datasets/CAER-S/CAER-S-processed/test_image.pkl', 'rb') as f:
            self.image_ = pickle.load(f)

        self.mode = mode
        self.img_size = img_size
        self.use_all = use_all

    def precoess(self):
        data = []

        for k in self.info.keys():
            seqs = self.info[k]

            L = len(seqs)
            if L > 1:
                indices = np.random.choice(
                    range(L), size=L if self.use_all else 1, replace=False)
                seq = [seqs[idx] for idx in indices]
            else:
                seq = seqs
            data += seq
        random.shuffle(data)
        self.data = data

    def get_transform_simple(self, flip_p=False):
        if self.mode in ['train', 'trainval']:
            transform_List = [transforms.ToPILImage(), ]
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            transform_List += [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                               # transforms.RandomErasing(p=0.4, value=[0.0, 0.0, 0.0])
                               ]  # 0.4 0.4 0.4

            self.transform = transforms.Compose(transform_List)
        else:
            flip_p = False
            transform_List = [transforms.ToPILImage()]
            self.transform = transforms.Compose(transform_List)

        return self.transform, flip_p

    def get_transform(self, flip_p=False, crop=224, RE=[0.0, 0.0, 0.0], colorjitter=None):
        if self.mode in ['train', 'trainval']:
            transform_List = [transforms.ToPILImage()]
            if crop is not None:
                transform_List.append(transforms.RandomCrop(size=crop))
            if flip_p:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__flip__(img, flip_p)))
            if colorjitter is not None:
                transform_List.append(transforms.Lambda(
                    lambda img: self.__jitter__(img, colorjitter)))
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            if RE is not None:
                transform_List.append(
                    transforms.RandomErasing(p=0.4, value=RE))
            self.transform = transforms.Compose(transform_List)
        else:
            transform_List = [transforms.ToPILImage()]
            transform_List += [transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
            self.transform = transforms.Compose(transform_List)

        return self.transform

    @staticmethod
    def __flip__(img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def __jitter__(img, params):
        brightness, contrast, saturation, hue = params
        if isinstance(img, np.ndarray):

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage(
            )] + img_transforms + [np.array, img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                jittered_img = img
                for func in img_transforms:
                    jittered_img = func(jittered_img)
                jittered_img = jittered_img.astype('float32')

        elif isinstance(img, Image.Image):
            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(
                    lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all images
            jittered_img = img
            for func in img_transforms:
                jittered_img = func(jittered_img)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img)))
        return jittered_img

    @staticmethod
    def get_jitter_params(brightness=0, contrast=0, saturation=0, hue=0):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None

        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    @staticmethod
    def adjust_bbox(im_size, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(im_size[1], max(0, x1))
        x2 = min(im_size[1], max(0, x2))
        y1 = min(im_size[0], max(0, y1))
        y2 = min(im_size[0], max(0, y2))

        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def coord_origin_translation(coord, x0, y0):
        x1, y1, x2, y2 = coord

        return (x1 - x0, y1 - y0, x2 - x0, y2 - y0)

    @staticmethod
    def scale_coord(coord, src, tgt):
        x1, y1, x2, y2 = coord
        h0, w0 = src
        h1, w1 = tgt

        x1 *= w1 / w0
        y1 *= h1 / h0
        x2 *= w1 / w0
        y2 *= h1 / h0

        return (int(x1), int(y1), int(x2), int(y2))

    @staticmethod
    def enlarge_bbox(coord, scale=[0.8, 1.2]):
        x1, y1, x2, y2 = coord
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        r = random.uniform(scale[0], scale[1])
        h = (y2 - y1) / 2 * r
        w = (x2 - x1) / 2 * r

        xx1 = int(cx - w)
        yy1 = int(cy - h)
        xx2 = int(cx + w) if int(cx + w) > xx1 else xx1 + 1
        yy2 = int(cy + h) if int(cy + h) > yy1 else yy1 + 1

        return [xx1, yy1, xx2, yy2]

    def random_crop(self, image, body_coord, head_coord):
        if self.mode in ['train', 'trainval']:
            h, w, _ = image.shape
            x1, y1, x2, y2 = body_coord

            bg_x1 = random.randint(0, x1)
            bg_y1 = random.randint(0, y1)
            bg_x2 = random.randint(x2, w)
            bg_y2 = random.randint(y2, h)

            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

        return image, body_coord, head_coord

    def random_crop2(self, image, body_coord, head_coord):
        if self.mode != 'test':
            h, w, _ = image.shape
            x01, y01, x02, y02 = head_coord
            x11, y11, x12, y12 = body_coord

            x21 = random.randint(min(x01, x11), x01)
            y21 = random.randint(min(y01, y11), y01)
            x22 = random.randint(x02, max(x02, x12))
            y22 = random.randint(y02, max(y02, y12))

            bg_x1 = random.randint(0, x21)
            bg_y1 = random.randint(0, y21)
            try:
                bg_x2 = random.randint(x22, w)
            except:
                print(w-1, w)
                bg_x2 = w-1
            bg_y2 = random.randint(y22, h)

            body_coord = (x21, y21, x22, y22)
            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

            body_coord = self.adjust_bbox(image.shape[:2], body_coord)
            head_coord = self.adjust_bbox(image.shape[:2], head_coord)

        return image, body_coord, head_coord

    def random_crop3(self, image, body_coord, head_coord):
        if self.mode in ['train', 'trainval']:
            h, w, _ = image.shape

            head_coord = self.adjust_bbox(
                [h, w], self.enlarge_bbox(head_coord, scale=[0.8, 1.2]))
            body_coord = self.adjust_bbox(
                [h, w], self.enlarge_bbox(body_coord, scale=[1.2, 1.2]))

            x01, y01, x02, y02 = head_coord
            x11, y11, x12, y12 = body_coord

            x21 = random.randint(min(x01, x11), x01)
            y21 = random.randint(min(y01, y11), y01)
            x22 = random.randint(x02, max(x02, x12))
            y22 = random.randint(y02, max(y02, y12))

            bg_x1 = random.randint(0, x21)
            bg_y1 = random.randint(0, y21)
            bg_x2 = random.randint(x22, w)
            bg_y2 = random.randint(y22, h)

            body_coord = (x21, y21, x22, y22)
            image = copy.deepcopy(image[bg_y1: bg_y2, bg_x1: bg_x2])
            body_coord = self.coord_origin_translation(
                body_coord, bg_x1, bg_y1)
            head_coord = self.coord_origin_translation(
                head_coord, bg_x1, bg_y1)

            body_coord = self.adjust_bbox(image.shape[:2], body_coord)
            head_coord = self.adjust_bbox(image.shape[:2], head_coord)

        return image, body_coord, head_coord

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, cat_label, _, body_coord, head_coord = self.data[index]
        image = self.image[key]

        image, body_coord, head_coord = self.random_crop2(
            image, body_coord, head_coord)
        H, W, _ = image.shape

        flip_p = random.random() < 0.5
        trans, flip_p = self.get_transform_simple(flip_p)
        ctx_info = trans(image)
        if flip_p:
            body_coord = (W - body_coord[2], body_coord[1],
                          W - body_coord[0], body_coord[3])
            head_coord = (W - head_coord[2], head_coord[1],
                          W - head_coord[0], head_coord[3])
        body_info = copy.deepcopy(ctx_info.crop(body_coord))
        head_info = copy.deepcopy(ctx_info.crop(head_coord))
        cat_info = torch.tensor(cat_label, dtype=torch.float32)

        trans_base = transforms.Compose([transforms.Resize(size=(224, 224)),  # 224,224
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                                         # transforms.RandomErasing(p=0.4, value=[0.0, 0.0, 0.0])
                                         ])  # 2024 -> resize

        ctx_img = trans_base(ctx_info)
        body_img = trans_base(body_info)
        head_img = trans_base(head_info)

        body_coord = self.scale_coord(
            body_coord, (H, W), (self.img_size, self.img_size))
        head_coord = self.scale_coord(
            head_coord, (H, W), (self.img_size, self.img_size))

        # --- debug ---
        # cv2.imwrite('./emotion/train_transimg/ctx_img.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # cv2.waitKey() image.permute(1,2,0).numpy() cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('./emotion/train_transimg/body_img.jpg', cv2.cvtColor(image[body_coord[1]:body_coord[3],body_coord[0]:body_coord[2]], cv2.COLOR_BGR2RGB))
        # cv2.waitKey()
        # cv2.imwrite('./emotion/train_transimg/head_img.jpg', cv2.cvtColor(head_info, cv2.COLOR_BGR2RGB))
        # cv2.waitKey()

        return ctx_img, body_img, head_img, cat_info, torch.tensor(body_coord), torch.tensor(head_coord)
