import os
import numpy as np
from PIL import Image
def save_image(image_dict_list, opt, epoch):
    #epoch被整除时保存图片
    if epoch % 5 != 0:
        return
    save_path = opt.savepath + '\\' + 'Freeman_Dbl_Red' + '\\' + 'epoch_' + str(epoch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i,dict in enumerate(image_dict_list):
        image = Image.fromarray(dict['image'].transpose(1,2,0)).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'image.png')
        label = Image.fromarray(dict['label']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'label.png')
        # Freeman_Odd_Blue = Image.fromarray(dict['Freeman_Odd_Blue']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'Freeman_Odd_Blue.png')
        # Freeman_Dbl_Red = Image.fromarray(dict['Freeman_Dbl_Red']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'Freeman_Dbl_Red.png')
        # Freeman_Vol_Green = Image.fromarray(dict['Freeman_Vol_Green']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'Freeman_Vol_Green.png')

        mask = Image.fromarray(dict['mask']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'mask.png')
        cam = Image.fromarray(dict['cam']).save(save_path + '\\' + 'batch_' + str(i) + '_' + 'cam.png')
    print('save image success')


def save_test_image(test_image_dict_list,opt):

    save_path_combined = opt.test_result_path + '\\' + opt.arch + '\\' + 'combined'
    save_path_mask = opt.test_result_path + '\\' + opt.arch + '\\' + 'mask'
    if not os.path.exists(save_path_combined):
        os.makedirs(save_path_combined)
    if not os.path.exists(save_path_mask):
        os.makedirs(save_path_mask)
    for i, bacth in enumerate(test_image_dict_list):
        for j in range(bacth['image'].shape[0]):

            image = Image.fromarray(bacth['image'][j])
            label = Image.fromarray(bacth['label'][j])
            mask = Image.fromarray(bacth['mask'][j])

            width, height = bacth['image'][j].shape[0], bacth['image'][j].shape[1]
            combined_width = width * 3
            combined_height = height
            combined_image = Image.new('RGB', (combined_width, combined_height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(label, (width, 0))
            combined_image.paste(mask, (width * 2, 0))
            combined_image.save(save_path_combined + '/' + 'batch_' + str(i) + '_' + 'combined' + str(j) + '.png')

            mask.save(save_path_mask + '/' + 'batch_' + str(i) + '_' + 'mask' + str(j) + '.png')
