from skimage.transform import resize
import numpy as np
import os
import SimpleITK as sitk
import h5py

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, \
        "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] >= -325
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

# 获取bounding box的函数。这个在日常会经常用到
def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

# 根据bbox提取ROI，这个也经常会用到
def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def _compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    median = np.median(voxels)
    mean = np.mean(voxels)
    sd = np.std(voxels)
    mn = np.min(voxels)
    mx = np.max(voxels)
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5


def resampleVolume(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
    outsize = [outsize[2], outsize[1], outsize[0]]

    # 设定重采样的一些参数
    image_data = sitk.GetArrayFromImage(vol)
    im = resize(image_data, output_shape=outsize, order=3, mode='edge', anti_aliasing=False)
    image = sitk.GetImageFromArray(im)
    image.SetSpacing(outspacing)
    image.SetDirection(vol.GetDirection())
    image.SetOrigin(vol.GetOrigin())

    return image


def resampleMask(outspacing, vol):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
    outsize = [outsize[2], outsize[1], outsize[0]]

    # 设定重采样的一些参数
    image_data = sitk.GetArrayFromImage(vol)
    im = resize(image_data, output_shape=outsize, order=0, anti_aliasing=False)
    image = sitk.GetImageFromArray(im)
    image.SetSpacing(outspacing)
    image.SetDirection(vol.GetDirection())
    image.SetOrigin(vol.GetOrigin())

    return image

def normalize_data(data):  # flare
    data = np.clip(data, -22.0, 325.0)
    data = np.array(data, dtype=np.float32)
    data -= 214.68231
    data /= 100.240135
    return data

def resample_label_image(label_image, new_spacing, interpolator=sitk.sitkNearestNeighbor):
    original_spacing = label_image.GetSpacing()
    original_size = label_image.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(label_image.GetDirection())
    resample.SetOutputOrigin(label_image.GetOrigin())
    resample.SetInterpolator(interpolator)

    return resample.Execute(label_image)



imdir = "F:/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/imagesTr/"   # DIRECTORY PATH TO data
labeldir = "F:/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/labelsTr/"  # DIRECTORY PATH TO label
outdir = "F:/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/h555/imageout/"  # DIRECTORY PATH TO processed data nii
outdir2 = "F:/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/h555/labelout//" # DIRECTORY PATH TO processed label nii
out_dir = "F:/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/h555/h5/"  # DIRECTORY PATH TO processed h5


for pdx, fname in enumerate(sorted(getFiles(imdir))):
    sitk_im = sitk.ReadImage(os.path.join(imdir, fname))
    fname = fname.replace('_0000', '')
    sitk_mask = sitk.ReadImage(os.path.join(labeldir, fname))

    img_fdata = sitk.GetArrayFromImage(sitk_im)
    img_fdata = img_fdata.astype(np.float32)
    mask_fdata = sitk.GetArrayFromImage(sitk_mask)

    directions = np.asarray(sitk_im.GetDirection())
    directions = directions.tolist()
    if directions == [1, 0, 0, 0, -1, 0, 0, 0, -1]:

        img_fdata = np.flip(img_fdata, [0, 2])
        mask_fdata = np.flip(mask_fdata, [0, 2])

        data = np.clip(img_fdata, -325, 325)

        saveimg = sitk.GetImageFromArray(data)
        saveimg.SetSpacing(sitk_im.GetSpacing())
        saveimg.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        saveimg.SetOrigin(sitk_im.GetOrigin())

        vol_resampled = resampleVolume([1.22250766, 1.22250766, 2.5], saveimg) #flare

        savemask = sitk.GetImageFromArray(mask_fdata)
        savemask.SetSpacing(sitk_im.GetSpacing())
        savemask.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
        savemask.SetOrigin(sitk_im.GetOrigin())

        vol_resampled_mask = resampleMask([1.22250766, 1.22250766, 2.5], savemask) #flare

        resize_imgarr = sitk.GetArrayFromImage(vol_resampled)  # 96 128 128
        nor_resize_imgarr = normalize_data(resize_imgarr)
        nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
        nor_resize_img.SetSpacing(vol_resampled.GetSpacing())
        nor_resize_img.SetOrigin(vol_resampled.GetOrigin())
        nor_resize_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])

        print(nor_resize_img.GetDirection())

        sitk.WriteImage(nor_resize_img, outdir + fname.split('.')[0] + '.nii.gz')

        sitk.WriteImage(vol_resampled_mask, outdir2 + fname.split('.')[0] + '.nii.gz')

        #  to h5
        savename = out_dir + fname.split('.')[0]
        if not os.path.isdir(savename):
            os.makedirs(savename)

        im = sitk.GetArrayFromImage(nor_resize_img)
        label = sitk.GetArrayFromImage(vol_resampled_mask)
        directions = np.asarray(nor_resize_img.GetDirection())

        f = h5py.File(savename + '/2022.h5', 'w')
        f.create_dataset('image', data=im, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()


    else:

        data = np.clip(img_fdata, -325, 325)

        saveimg = sitk.GetImageFromArray(img_fdata)
        saveimg.SetSpacing(sitk_im.GetSpacing())
        saveimg.SetDirection(sitk_im.GetDirection())
        saveimg.SetOrigin(sitk_im.GetOrigin())

        vol_resampled = resampleVolume([1.22250766, 1.22250766, 2.5], saveimg)  #flare

        savemask = sitk.GetImageFromArray(mask_fdata)
        savemask.SetSpacing(sitk_im.GetSpacing())
        savemask.SetDirection(sitk_im.GetDirection())
        savemask.SetOrigin(sitk_im.GetOrigin())

        vol_resampled_mask = resampleMask([1.22250766, 1.22250766, 2.5], savemask) #flare

        resize_imgarr = sitk.GetArrayFromImage(vol_resampled)  # 96 128 128
        nor_resize_imgarr = normalize_data(resize_imgarr)
        nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
        nor_resize_img.SetSpacing(vol_resampled.GetSpacing())
        nor_resize_img.SetOrigin(vol_resampled.GetOrigin())
        nor_resize_img.SetDirection(vol_resampled.GetDirection())

        print(nor_resize_img.GetDirection())

        sitk.WriteImage(nor_resize_img, outdir + fname.split('.')[0] + '.nii.gz')

        sitk.WriteImage(vol_resampled_mask, outdir2 + fname.split('.')[0] + '.nii.gz')

        #  to h5
        savename = out_dir + fname.split('.')[0]
        if not os.path.isdir(savename):
            os.makedirs(savename)

        im = sitk.GetArrayFromImage(nor_resize_img)
        label = sitk.GetArrayFromImage(vol_resampled_mask)
        directions = np.asarray(nor_resize_img.GetDirection())

        f = h5py.File(savename + '/2022.h5', 'w')
        f.create_dataset('image', data=im, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
