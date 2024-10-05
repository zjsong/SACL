"""
A base class for constructing PyTorch AudioVisual dataset.
"""


import random
import librosa
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from skimage.segmentation import felzenszwalb

import torch
import torch.utils.data as torchdata
import torchvision.transforms.functional as TF
from torchvision import transforms

from utils import normalize_audio


class BaseDataset(torchdata.Dataset):
    def __init__(self, args, mode='train'):

        self.args = args
        self.mode = mode
        self.seed = args.seed
        random.seed(self.seed)

        self.imgSize = args.imgSize
        self.audRate = args.audRate
        self.audSec = args.audSec  # 3s
        self.audLen = args.audRate * args.audSec
        self.trainset = args.trainset
        self.testset = args.testset
        self.gridSize = args.gridSize
        self.blockSize = args.imgSize // args.gridSize

        if self.mode == 'train':
            self.num_data = args.num_train
            print('number of training samples: ', self.num_data)

        elif self.mode == 'test':
            if self.testset == 'flickr':
                data_path = args.data_path + 'SoundNet_Flickr/flickr_test249_in5k.csv'
            elif self.testset == 'vggss':
                data_path = args.data_path + 'VGG-Sound/vggss_test_4692.csv'

            self.data_ids = pd.read_csv(data_path, header=None, sep=',')
            self.num_data = self.data_ids.shape[0]
            print('number of test samples: ', self.num_data)

    def __len__(self):
        return self.num_data

    def _load_frame_mask(self, frame=None, path=None, mode='train', num_trans=2):
        """
        Generate num_trans samples and corresponding masks with the same augmentation.
        """

        # load frame
        if frame is not None:
            frame = Image.fromarray(np.uint8(frame))
        else:
            frame = Image.open(path).convert('RGB')

        # generate mask
        if self.gridSize < 0:   # generate pseudo mask with the unsupervised method (FH)
            mask = felzenszwalb(np.array(frame), scale=1000, sigma=0.5, min_size=1000)

        else:   # generate grid pseudo mask: 1 x 1, 2 x 2, 4 x 4, 8 x 8
            mask = np.zeros((self.imgSize, self.imgSize))
            rand_label = np.random.permutation(self.gridSize * self.gridSize)
            for i, label in enumerate(rand_label):
                row = i // self.gridSize
                col = i % self.gridSize
                mask[row*self.blockSize:(row+1)*self.blockSize, col*self.blockSize:(col+1)*self.blockSize] = label

        mask = Image.fromarray(np.uint8(mask))

        tran_frame = [0] * num_trans
        tran_mask = [0] * num_trans
        for i in range(num_trans):
            tran_frame[i], tran_mask[i] = self.tran_image_mask(frame, mask, mode)

        frame_orig, mask_orig = self.tran_image_mask(frame, mask, mode='test')
        tran_frame.append(frame_orig)
        tran_mask.append(mask_orig)

        return tran_frame, tran_mask

    def tran_image_mask(self, image, mask, mode='train'):
        
        if mode == 'train':

            # RandomResizedCrop -- image & mask
            resize = transforms.Resize(size=(int(self.imgSize * 1.1), int(self.imgSize * 1.1)), interpolation=Image.BICUBIC)
            image = resize(image)
            mask = resize(mask)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.imgSize, self.imgSize))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # RandomHorizontalFlip -- image & mask
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random ColorJitter -- image
            if random.random() > 0.2:
                colorjitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                image = colorjitter(image)

            # RandomGrayscale -- image
            if random.random() > 0.8:
                image = TF.to_grayscale(image, num_output_channels=3)

            # Random GaussianBlur -- image
            if random.random() > 0.5:
                gaussianblur = GaussianBlur([.1, 2.])
                image = gaussianblur(image)

            # To Tensor -- image & mask
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            # Normalize -- image
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        else:

            # Resize -- image & mask
            resize = transforms.Resize(size=(self.imgSize, self.imgSize), interpolation=Image.BICUBIC)
            image = resize(image)
            mask = resize(mask)

            # To Tensor -- image & mask
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

            # Normalization -- image
            image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, (mask * 255).type(torch.uint8)

    def _load_audio(self, path):
        """
        Load wav file.
        """

        # load audio
        audio_np, rate = librosa.load(path, sr=self.audRate, mono=True)

        curr_audio_length = audio_np.shape[0]
        if curr_audio_length < self.audLen:
            n = int(self.audLen / curr_audio_length) + 1
            audio_np = np.tile(audio_np, n)
            curr_audio_length = audio_np.shape[0]

        start_sample = int(curr_audio_length / 2) - int(self.audLen / 2)
        audio_np = normalize_audio(audio_np[start_sample:start_sample + self.audLen])

        return audio_np


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
