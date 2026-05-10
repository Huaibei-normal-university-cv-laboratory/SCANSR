import os
import random
import SimpleITK as sitk
import cv2
import numpy as np
import torch
from tqdm import tqdm

basePath = '/home/cjc/cwj/dataSet/ixi'
slice_volume_per = 50
pd = f'{basePath}/IXI-PD/'
t2 = f'{basePath}/IXI-T2/'
train_HR_PD = f'{basePath}/train_HR_PD'
train_HR_T2 = f'{basePath}/train_HR_T2'
train_LR_T2 = f'{basePath}/train_LR_T2'
val_HR_PD = f'{basePath}/val_HR_PD'
val_HR_T2 = f'{basePath}/val_HR_T2'
val_LR_T2 = f'{basePath}/val_LR_T2'
test_HR_PD = f'/{basePath}/test_HR_PD'
test_HR_T2 = f'{basePath}/test_HR_T2'
test_LR_T2 = f'{basePath}/test_LR_T2'
pds = os.listdir(pd)
t2s = os.listdir(t2)
random.seed(1)
random.shuffle(t2s)
os.makedirs(train_HR_PD, exist_ok=True)
os.makedirs(train_HR_T2, exist_ok=True)
os.makedirs(val_HR_T2, exist_ok=True)
os.makedirs(val_HR_PD, exist_ok=True)
os.makedirs(test_HR_T2, exist_ok=True)
os.makedirs(test_HR_PD, exist_ok=True)
os.makedirs(os.path.join(train_LR_T2,'X2'),exist_ok=True)
os.makedirs(os.path.join(train_LR_T2,'X4'),exist_ok=True)
os.makedirs(os.path.join(val_LR_T2,'X2'),exist_ok=True)
os.makedirs(os.path.join(val_LR_T2,'X4'),exist_ok=True)
os.makedirs(os.path.join(test_LR_T2,'X2'),exist_ok=True)
os.makedirs(os.path.join(test_LR_T2,'X4'),exist_ok=True)
# train 400
# val 20
# test 80
# normalize to 0-255
def norm(data):
    data = data.astype(np.float32)
    # data = np.clip(data, a_min=-200, a_max=400)
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data * 255.

def generate_unique_numbers(start, end, count):
    if end - start + 1 < count:
        raise ValueError("范围太小，无法生成足够不重复的数字")
    numbers = set()
    while len(numbers) < count:
        number = random.randint(start, end)
        numbers.add(number)
    return list(numbers)


def saveToImage(nii, filename, savedir, slice):
    if 'IXI014-HH' in filename:
        print('====IXI014-HH ')
        print('====IXI014-HH ')
        return
    img_1 = sitk.ReadImage(nii, sitk.sitkInt16)
    space = img_1.GetSpacing()
    img_1 = sitk.GetArrayFromImage(img_1)
    width, height, queue = img_1.shape
    data_1 = norm(img_1)
    total_number = 0
    for i in slice:
        total_number = total_number + 1
        img_arr1 = data_1[i, :, :]
        img_arr1 = np.expand_dims(img_arr1, axis=2)
        cv2.imwrite(savedir + '/{}_{:02d}.png'.format(filename, total_number), img_arr1)

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))
def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    # print(data.shape)
    # print(data.shape[-2],data.shape[-1],data.shape[0],data.shape[1])
    assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)  # 556...556
    assert 0 < shape[1] <= data.shape[-1]  # 640...640
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def saveLR(hr,savePath,scale):
    im1_GT = cv2.imread(hr, cv2.IMREAD_UNCHANGED)
    filename = hr.split('/')[-1]
    im1_GT = torch.tensor(im1_GT).unsqueeze(0).float() / 255.
    hr_data = im1_GT
    imgfft = fft2(hr_data)

    imgfft = center_crop(imgfft, (256//scale, 256//scale))
    t = np.fft.ifft2(imgfft)
    LR_ori = np.abs(t)
    t2_image = cv2.normalize(LR_ori, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to uint8
    t2_image = np.uint8(t2_image)
    path = f'{savePath}/X{scale}/{filename}'
    cv2.imwrite(path,t2_image[0])

def generateLR(HR_path,LR_path):
    for file in tqdm(os.listdir(HR_path)):
        saveLR(os.path.join(HR_path,file),LR_path,2)
        saveLR(os.path.join(HR_path,file),LR_path,4)





def generateImage():
    # create Dataset train hr PD T2
    print('create Dataset train hr')
    for file in tqdm(t2s[0:401]):
        # print('starting %s' % file)
        t2_file_path = os.path.join(t2, file)
        pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
        slice_index = generate_unique_numbers(0, 110, slice_volume_per)
        saveToImage(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), train_HR_T2, slice_index)
        saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), train_HR_PD, slice_index)
    generateLR(train_HR_T2, train_LR_T2)
    # val
    print('create Dataset val hr')
    for file in tqdm(t2s[402:422]):
        # print('starting %s' % file)
        t2_file_path = os.path.join(t2, file)
        pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
        slice_index = generate_unique_numbers(0, 110, slice_volume_per)
        saveToImage(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), val_HR_T2, slice_index)
        saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), val_HR_PD, slice_index)
    generateLR(val_HR_T2, val_LR_T2)
    # test
    print('create Dataset test hr')
    for file in tqdm(t2s[423:503]):
        # print('starting %s' % file)
        t2_file_path = os.path.join(t2, file)
        pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
        slice_index = generate_unique_numbers(0, 110, slice_volume_per)
        saveToImage(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), test_HR_T2, slice_index)
        saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), test_HR_PD, slice_index)
    generateLR(test_HR_T2, test_LR_T2)


def ttt():
    # 研究数据
    def saveToImage2(nii, filename, savedir):
        if 'IXI014-HH' in filename:
            print('====IXI014-HH ')
            print('====IXI014-HH ')
            return
        img_1 = sitk.ReadImage(nii, sitk.sitkInt16)
        space = img_1.GetSpacing()
        img_1 = sitk.GetArrayFromImage(img_1)
        width, height, queue = img_1.shape
        data_1 = norm(img_1)
        total_number = 0
        nii_path = os.path.join(savedir,filename)
        os.makedirs(nii_path,exist_ok=True)
        for i in range(width):
            total_number = total_number + 1
            img_arr1 = data_1[i, :, :]
            img_arr1 = np.expand_dims(img_arr1, axis=2)
            cv2.imwrite(nii_path + '/{:02d}.png'.format(total_number), img_arr1)

    os.makedirs(f'{basePath}/train_HR_T2_study_data',exist_ok=True)
    for file in tqdm(t2s[0:100]):
        # print('starting %s' % file)
        t2_file_path = os.path.join(t2, file)
        # pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
        # slice_index = generate_unique_numbers(0, 110, slice_volume_per)
        saveToImage2(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), f'{basePath}/train_HR_T2_study_data')
        # saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), train_HR_PD, slice_index)


# generateImage()


if __name__ == '__main__':
    generateImage()
    # # val
    # print('create Dataset val hr')
    # for file in tqdm(t2s[400:420]):
    #     # print('starting %s' % file)
    #     t2_file_path = os.path.join(t2, file)
    #     pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
    #     slice_index = generate_unique_numbers(0, 110, slice_volume_per)
    #     saveToImage(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), val_HR_T2, slice_index)
    #     saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), val_HR_PD, slice_index)
    # generateLR(val_HR_T2, val_LR_T2)
    # # test
    # print('create Dataset test hr')
    # for file in tqdm(t2s[420:500]):
    #     # print('starting %s' % file)
    #     t2_file_path = os.path.join(t2, file)
    #     pd_file_path = os.path.join(pd, file.replace('T2', 'PD'))
    #     slice_index = generate_unique_numbers(0, 110, slice_volume_per)
    #     saveToImage(t2_file_path, t2_file_path.split('/')[-1].strip('.nii.gz'), test_HR_T2, slice_index)
    #     saveToImage(pd_file_path, pd_file_path.split('/')[-1].strip('.nii.gz'), test_HR_PD, slice_index)
    # generateLR(test_HR_T2, test_LR_T2)