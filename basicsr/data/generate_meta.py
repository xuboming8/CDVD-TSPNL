import os
path = '/media/xbm/data/VideoDeblur_Dataset/RealBlur_padding/RealBlur_R/train/gt'
f = open('/media/xbm/data/xbm/BasicSR/basicsr/data/meta_info/meta_info_REALR_GT.txt',"w")
filelist = os.listdir(path)
filelist.sort()
for file in filelist:
    path_file = path + '/' + file
    image_list = os.listdir(path_file)
    image_list.sort()
    i = 0
    for image in image_list:
        i += 1
    line = file + ' ' + str(i) + ' ' + '(772,676,3)'
    f.write(line + '\n')
f.close()