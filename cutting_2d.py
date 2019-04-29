from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import os,config
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
tqdm = lambda x: x


def load_itkfilewithtrucation(filename, upper=200, lower=-200):
    """
    load mhd files,set truncted value range and normalization 0-255
    :param filename:
    :param upper:
    :param lower:
    :return:
    """
    # 1,tructed outside of liver value
    srcitkimage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
    srcitkimagearray = sitk.GetArrayFromImage(srcitkimage)
    srcitkimagearray[srcitkimagearray > upper] = upper
    srcitkimagearray[srcitkimagearray < lower] = lower
    # 2,get tructed outside of liver value image
    sitktructedimage = sitk.GetImageFromArray(srcitkimagearray)
    origin = np.array(srcitkimage.GetOrigin())
    spacing = np.array(srcitkimage.GetSpacing())
    sitktructedimage.SetSpacing(spacing)
    sitktructedimage.SetOrigin(origin)
    # 3 normalization value to 0-255
    rescalFilt = sitk.RescaleIntensityImageFilter()
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    itkimage = rescalFilt.Execute(sitk.Cast(sitktructedimage, sitk.sitkFloat32))
    return itkimage


# Some helper functions

def get_cube_from_img(img3d, center, block_size):
    # get roi(z,y,x) image and in order the out of img3d(z,y,x)range
    start_z = center[0]
    center_y = center[1]
    center_x = center[2]
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size
    start_y = max(center_y - block_size / 2, 0)
    if start_y + block_size > img3d.shape[1]:
        start_y = img3d.shape[1] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    roi_img3d = img3d[start_z:start_z+1, start_y:start_y + block_size, start_x:start_x + block_size]
    return roi_img3d

# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return (f)


def get_node_classify():
    # Getting list of image files and output nodule 0 and 1
    for subsetindex in range(1):
        classify_size = 48
        luna_path = config.path_to_subsets
        luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
        output_path = config.path_to_cls_img
        file_list = glob(luna_subset_path + "*.mhd")

        # The locations of the nodes
        luna_csv_path = config.path_to_info
        df_node = pd.read_csv(luna_csv_path + "final_candidates.csv")
        df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
        df_node = df_node.dropna()
        # Looping over the image files
        for fcount, img_file in enumerate(tqdm(file_list)):
            # get all nodules associate with file
            mini_df = df_node[df_node["file"] == img_file]
            # some files may not have a nodule--skipping those
            if mini_df.shape[0] > 0:
                # load the data once
                itk_img = load_itkfilewithtrucation(img_file, 600, -1000)
                img_array = sitk.GetArrayFromImage(itk_img)
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                print(origin)
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                print(origin)
                # go through all nodes
                index = 0
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    label = cur_row["class"]
                    # nodule center
                    center = np.array([node_x, node_y, node_z])
                    # nodule center in voxel space (still x,y,z ordering)  # clip prevents going out of bounds in Z
                    if label == 0:
                        v_center = np.rint((center - origin) / spacing)
                    else:
                        v_center = np.rint(center)
                        v_center[2] = np.rint((center[2] - origin[2]) / spacing[2])
                    # convert x,y,z order v_center to z,y,x order v_center
                    v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                    # get cub size of classify_size
                    node_cube = get_cube_from_img(img_array, v_center, classify_size)
                    node_cube.astype(np.uint8)
                    print(node_cube[0].max())
                    if label!=0:
                        plt.imshow(node_cube[0])
                    plt.show()
                    # save as bmp file
                    # for i in range(classify_size):
                    #     if label == 1:
                    #         filepath = output_path + "1/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    #     if label == 0:
                    #         filepath = output_path + "0/" + str(subsetindex) + "_" + str(fcount) + "_" + str(index) + "/"
                    #         if not os.path.exists(filepath):
                    #             os.makedirs(filepath)
                    #         cv2.imwrite(filepath + str(i) + ".bmp", node_cube[i])
                    # index += 1
                    # save as npy file
                    filepath = output_path + str(label) + "/"
                    if not os.path.exists(filepath):
                        os.makedirs(filepath)
                    filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                    np.save(filepath + filename + ".npy", node_cube[0])

                    #if label == 1:
                    #    filepath = output_path + "1/"
                    #    if not os.path.exists(filepath):
                    #        os.makedirs(filepath)
                    #    filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                    #    np.save(filepath + filename + ".npy", node_cube)
                    #if label == 0:
                    #    filepath = output_path + "0/"
                    #    if not os.path.exists(filepath):
                    #        os.makedirs(filepath)
                    #    filename = str(subsetindex) + "_" + str(fcount) + "_" + str(index)
                    #    np.save(filepath + filename + ".npy", node_cube)
                    index += 1


get_node_classify()