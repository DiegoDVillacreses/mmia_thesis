import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import pandas as pd
import numpy as np
from PIL import Image
from utils import cropp
import torchvision.transforms as transforms
from src.config import TARGET_SIZE, PATH_UNLABELED_METADATA

import json

def load_labeled_data(dir_data):

    # Get image and label file name
    train_images = os.listdir(dir_data + "/train/images")
    valid_images = os.listdir(dir_data + "/valid/images")
    test_images = os.listdir(dir_data + "/test/images")

    train_labels = os.listdir(dir_data + "/train/labels")
    valid_labels = os.listdir(dir_data + "/valid/labels")
    test_labels = os.listdir(dir_data + "/test/labels")

    train = ['train' for i in train_images]
    valid = ['valid' for i in valid_images]
    test = ['test' for i in test_images]

    df1 = pd.DataFrame({"image_file": train_images + valid_images + test_images,
                        "subset": train + valid + test})
    df2 = pd.DataFrame({"image_label": train_labels + valid_labels + test_labels,
                        "subset": train + valid + test})

    df1['name'] = df1['image_file'].str.replace('.jpg','')
    df2['name'] = df2['image_label'].str.replace('.txt','')

    df = pd.merge(left = df1,
            right=df2,
            on = ['name','subset'],
            how = 'inner')
    del df1, df2

    # Load image and crop it
    l = []
    for i in range(df.shape[0]):
        image_file, subset, name, image_label = df.iloc[i, :]
        dir_x = dir_data + "/" + subset + "/" + '/labels/' + image_label
        with open(dir_x) as f:
            lines = f.readlines()
            if len(lines)>1:
                lines = [i.replace('\n','') for i in lines]
        image = Image.open(dir_data + "/" + subset + "/" + '/images/' + image_file)
        image_np = np.array(image)
        if len(lines)==1:
            line_x = [np.float64(i) for i in lines[0].split(' ')]
            image_cropped_np = np.array(cropp(line_x[1:], image))
            image_shape = ','.join([str(i) for i in image_cropped_np.shape])
            l.append([subset, image_file,int(line_x[0]),image_cropped_np,image_shape])
        if len(lines)>1:
            for line in lines:
                line_x = [np.float64(i) for i in line.split(' ')]
                image_cropped_np = np.array(cropp(line_x[1:], image))
                image_shape = ','.join([str(i) for i in image_cropped_np.shape])
                l.append([subset, name,int(line_x[0]),image_cropped_np,image_shape])
    df = pd.DataFrame(l, columns=['subset','file_name', 'label', 'cropped_image','image_shape'])
    df.loc[df['label']==0,'label_text'] = 'Taypec'
    df.loc[df['label']==1,'label_text'] = 'Taytaj'
    print(f"{df.shape=}")
    for i in range(df.shape[0]):
        ci_shape = [j for j in df.loc[i,'cropped_image'].shape]
        ci_px = ci_shape[0]*ci_shape[1]
        df.loc[i,'cropped_image_px'] = ci_px
    return df


def df_undersampling_strat(df, subset_col, label_col, random_state=0):
    """
    Balances the label_text within each subset by downsampling to the minimum count per subset.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - subset_col (str): The name of the subset column (e.g., 'test', 'train', 'valid').
    - label_col (str): The name of the label column to balance.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - pd.DataFrame: The balanced DataFrame.
    """
    min_counts = df.groupby(subset_col)[label_col].value_counts().groupby(level=0).min()
    # print("Minimum counts per subset:")
    # print(min_counts)
    # print("\n")
    def sample_group(group):
        subset = group.name[0]
        label = group.name[1]
        n = min_counts[subset]
        sampled = group.sample(n=n, random_state=random_state)
        return sampled
    balanced_df = df.groupby([subset_col, label_col]).apply(sample_group).reset_index(drop=True)
    return balanced_df


train_transform_labeled = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(size=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # If applicable
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value='random',
        inplace=False
    ),
])
val_transform_labeled = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])



train_transform_labeled_vit = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.RandomResizedCrop(size=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # If applicable
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10,
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value='random',
        inplace=False
    ),
])
val_transform_labeled_vit = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


def load_unlabeled_metadata():
    with open(os.path.join(PATH_UNLABELED_METADATA,'megadetector_metadata_modified_05_aug_2024.json'), 'r') as file:
        metadata = json.load(file)
    l_metadata = []
    for i in range(len(metadata)):
        mi = metadata[i]['detectors']['megadetectorV5']['output']
        if len(mi['detections'][0]['category'])>0:
            if mi['detections'][0]['category'][0]==0:
                l_metadata.append([mi['file'],len(mi['detections']), mi['detections'][0]['confidence'], mi['detections'][0]['bbox']])

    df_meta = pd.DataFrame(l_metadata,columns=['file', 'total_detections','confidence','bbox'])
    df_meta['len_bbox'] = [len(i) for i in df_meta['bbox']]
    df_meta['len_confidence'] = [len(i) for i in df_meta['confidence']]

    df_meta['confidence'] = [i[0] for i in df_meta['confidence']]
    df_meta['bbox'] = [i[0] for i in df_meta['bbox']]
    l = []
    for i in df_meta['file']:
        i = i.split("-")
        i = i[len(i)-1]
        l.append(i)
    df_meta['possible_animal_name']=l

    df_meta = df_meta[lambda x: x['confidence']>=0.85].reset_index(drop=True)

    files_list = df_meta['file'].to_list()
    bbox_list = df_meta['bbox'].to_list()
    return files_list, bbox_list

class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2, image_size = 96):
        self.base_transforms = base_transforms
        self.n_views = n_views
        self.image_size = image_size

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]
    

contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

val_transform_labeled_simclr = transforms.Compose([
    transforms.Lambda(lambda x: Image.fromarray(x)),
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]),
])