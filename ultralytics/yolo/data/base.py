# Ultralytics YOLO 🚀, AGPL-3.0 license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from ..utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM_BAR_FORMAT
from .utils import HELP_URL, IMG_FORMATS


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=None,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.im_files = self.get_img_files(self.img_path)
        if self.task_type == "multi":
            self.labels = self.get_multi_labels()
            self.update_multi_labels(include_class=classes)  # single_cls and include_class
            self.ni = len(self.labels[0])  # number of images
        else:
            self.labels = self.get_labels()
            self.update_labels(include_class=classes)  # single_cls and include_class
            self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Cache stuff
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims = [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)
        self.seg_transforms = self.build_seg_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def update_multi_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for task_labels in self.labels:  # 修改这里遍历每个任务的标签列表
            for i in range(len(task_labels)):  # 对每个任务标签列表进行遍历
                if include_class is not None:
                    cls = task_labels[i]['cls']
                    bboxes = task_labels[i]['bboxes']
                    segments = task_labels[i]['segments']
                    keypoints = task_labels[i]['keypoints']
                    j = (cls == include_class_array).any(1)
                    task_labels[i]['cls'] = cls[j]
                    task_labels[i]['bboxes'] = bboxes[j]
                    if segments:
                        task_labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                    if keypoints is not None:
                        task_labels[i]['keypoints'] = keypoints[j]
                if self.single_cls:
                    task_labels[i]['cls'][:, 0] = 0

    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz)),
                                interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(enumerate(results), total=self.ni, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        if self.task_type == "multi":
            bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
            nb = bi[-1] + 1  # number of batches
            s = np.array([x.pop('shape') for x in self.labels[0]])  # hw
            ar = s[:, 0] / s[:, 1]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            for label_index in range(len(self.labels)):
                self.labels[label_index] = [self.labels[label_index][i] for i in irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(
                int) * self.stride
            self.batch = bi  # batch index of image

        else:
            bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
            nb = bi[-1] + 1  # number of batches
            s = np.array([x.pop('shape') for x in self.labels])  # hw
            ar = s[:, 0] / s[:, 1]  # aspect ratio
            irect = ar.argsort()
            self.im_files = [self.im_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(
                int) * self.stride
            self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        ######
        if self.task_type == 'multi':
            label_list = self.get_label_info(index)
            transfer_lable = []

            # is_equal = np.array_equal(label_list[0]['img'], label_list[1]['img'])

            if self.augment:
                self.together = True  # the signal for augment
                transfer = self.transforms(label_list)
                self.together = False
                return transfer
            for i in range(len(label_list)):
                self.global_count = i

                if 'seg' in self.data['labels_list'][i]:
                    transfer_lable.append(self.seg_transforms(label_list[i]))
                    # transfer_lable.append(self.transforms(label_list[0]))
                else:
                    # test = self.transforms(label_list[i])
                    transfer_lable.append(self.transforms(label_list[i]))
            return transfer_lable
        ######
        else:
            label = self.get_label_info(index)
            # test = self.transforms(label)
            return self.transforms(label)

    def get_label_info(self, index):
        if self.task_type == 'multi':
            """Get and return label information from the dataset."""
            label_list = []
            for i in range(len(self.data['labels_list'])):
                label = deepcopy(
                    self.labels[i][index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
                label.pop('shape', None)  # shape is for rect, remove it
                label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
                label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                                      label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
                if self.rect:
                    label['rect_shape'] = self.batch_shapes[self.batch[index]]
                label = self.update_labels_info(label)
                label_list.append(label)
            return label_list
        else:
            """Get and return label information from the dataset."""
            label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
            label.pop('shape', None)  # shape is for rect, remove it
            label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
            label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                                  label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
            if self.rect:
                label['rect_shape'] = self.batch_shapes[self.batch[index]]
            label = self.update_labels_info(label)
            return label

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        if self.task_type == "multi":
            return len(self.labels[0])
        else:
            return len(self.labels)

    def update_labels_info(self, label):
        """custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError
