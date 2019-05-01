import os,config


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs


def save_file2csv(file_dir, file_name):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    out2 = open('ClassWeight.txt',"w")
    sub_dirs = file_name_path(file_dir)
    for subdir in sub_dirs:
        out2.write(str(len(os.listdir(file_dir+'/'+subdir)))+'\n')
        for file in os.listdir(file_dir+'/'+subdir):
            out.writelines(file_dir + "/" + subdir+'/'+file+'\n')



save_file2csv(config.path_to_cls_npy, "all.txt")
