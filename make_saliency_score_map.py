# import matplotlib.pyplot as plt

import numpy as np
import skimage.io as io
import pylab
import json
import cv2
import os
import time

a = time.time()

##################### crop imgs to 384 384 from directory  ##########################
def crop_384(_imgDir, _outDir):
    size = 384

    files = sorted(os.listdir(_imgDir))
    img_names = []
    for file in files:
        if file.endswith(".jpg"):
            img_names.append(file)
    i = 1
    for img_name in img_names:
        img_Dir= '{}/{}'.format(_imgDir, img_name)
        out_Dir = '{}/{}'.format(_outDir, img_name)
        img = cv2.imread(img_Dir)
        h, w, c = img.shape
        if min(h, w) >= size:
            cropped_img = img[h//2-192:h//2+192, w//2-192:w//2+192]
            cv2.imwrite(out_Dir, cropped_img)
            print("[" + str(i) + "] " + img_Dir+" >>>> Done")
        else:
            print("[" + str(i) + "] " + img_Dir + " >>>> "+"("+str(h)+", "+str(w)+")")

        i += 1

#####################################################################################


#####################################################################################
def get_json(_imgid, _phase):
    jsonFile = './json/{}2017_result_new.json'.format(_phase)

    with open(jsonFile) as json_file:
        json_data = json.load(json_file)
    return json_data

#####################################################################################

#####################################################################################
def get_ins_img(_imgDir, _json_sub_data):
    insDir = '{}/instance/'.format(_imgDir)
    #ratioDir = '{}/instance/ratio/'.format(_imgDir)
    # roiDir = '{}/instance/roi/'.format(_imgDir)

    img_info = _json_sub_data['image']
    img_name = img_info['name']
    key_list = list(_json_sub_data.keys())[1:]
    result = np.zeros((384, 384))
    for i in range(len(key_list)):
        instance_info = _json_sub_data[key_list[i]]
        if instance_info['roi_area'] != 0 :
            print("instance_info: ", instance_info)
            img_dir = '{}{}_{}.jpg'.format(insDir, img_name[:-4], instance_info['id'])

            _sub_img = cv2.imread(img_dir, 0) # 0: read img as grayscale
            h, w = _sub_img.shape
            if min(h, w) >= 384:
                cropped_img = _sub_img[h // 2 - 192:h // 2 + 192, w // 2 - 192:w // 2 + 192]
                cropped_bw = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_BINARY)[1] # numpy.ndarray

            if np.sum(cropped_bw)/255 != 0 :
                result = result + cropped_bw
                print(np.sum(result) / 255)
    return result
#####################################################################################

def get_ins_mask(_imgDir, _json_sub_data):
    insDir = '{}/instance/'.format(_imgDir)
    #ratioDir = '{}/instance/ratio/'.format(_imgDir)
    # roiDir = '{}/instance/roi/'.format(_imgDir)

    newDir = '{}_mask_ins/'.format(_imgDir)
    newDir_2 = '{}_mask_1st/'.format(_imgDir)
    img_info = _json_sub_data['image']
    img_name = img_info['name']
    key_list = list(_json_sub_data.keys())[1:]
    max_ins_id = ''
    result = np.zeros((384, 384))
    mask_1st = np.zeros((384, 384))

    for i in range(len(key_list)):
        instance_info = _json_sub_data[key_list[i]]
        if instance_info['roi_area'] != 0 :

            img_dir = '{}{}_{}.jpg'.format(insDir, img_name[:-4], instance_info['id'])

            _sub_img = cv2.imread(img_dir, 0) # 0: read img as grayscale
            h, w = _sub_img.shape
            if min(h, w) >= 384:
                cropped_img = _sub_img[h // 2 - 192:h // 2 + 192, w // 2 - 192:w // 2 + 192]
                cropped_bw = cv2.threshold(cropped_img, 127, 255, cv2.THRESH_BINARY)[1] # numpy.ndarray
                if max_ins_id == '':
                    max_ins_id = instance_info['id']
                    mask_1st = cropped_bw

                elif _json_sub_data[str(max_ins_id)]['instance_area'] < _json_sub_data[key_list[i]]['instance_area']:
                    max_ins_id = instance_info['id']
                    mask_1st = cropped_bw

            if np.sum(cropped_bw)/255 != 0 :
                result = result + cropped_bw

    if np.sum(result)/255 != 0 and np.sum(mask_1st)/255 != 0:
        cv2.imwrite(newDir + img_name, result)
        cv2.imwrite(newDir_2 + img_name, mask_1st)

    else:
        print(img_name)

    return result

def load_img_ids(imgDir):
    img_names = sorted(os.listdir(imgDir + '/'))
    img_ids = []
    for i in range (len(img_names)):
        #print(img_names[i])
        #if img_names[i] != "instance" or img_names[i] != "roi" or img_names[i] != "saliency":
        img_ids.append(int(img_names[i][:-4].lstrip('0')))
    print(len(img_ids))
    return img_ids

def val_remove():
    removeDir = './Dataset/COCO/val2017_crop'
    removeDir_2 = './Dataset/COCO/val2017_crop_LR_x4'
    remove_list_Dir = './Dataset/COCO/val2017_crop_mask/remove_list.txt'
    f = open(remove_list_Dir, 'r')
    i = 0
    while True:
        img_name = f.readline()
        if not img_name: break
        i += 1
        img_name = img_name[:-1] # discard \n
        print("[", i, "]", img_name)
        remove_img = '{}/{}'.format(removeDir, img_name)
        remove_img_2 = '{}/{}.png'.format(removeDir_2, img_name[:-4])
        if os.path.isfile(remove_img):
            os.remove(remove_img)
        if os.path.isfile(remove_img_2):
            os.remove(remove_img_2)
    f.close()


def make_score(_imgDir, _maskDir, _jsonDir):
    with open(jsonDir) as json_file:
        json_data = json.load(json_file)

    img_ids = load_img_ids(_imgDir)

    for i in range(len(img_ids)):
        # json_data[str(img_ids[i])]
        img_id = str(img_ids[i])
        total_ins_area = json_data[img_id]["image"]["total_ins_area"]
        # print(total_ins_area)
        instance_ids = [key for key in json_data[str(img_ids[i])]["instance"]]
        # print(instance_ids)
        scoreList = []

        for j in range (len(instance_ids)):
            ins_id = instance_ids[j]
            score = (json_data[img_id]["instance"][ins_id]["instance_area"] / total_ins_area + json_data[img_id]["instance"][ins_id]["roi_area/instance_area"])/2
            json_data[img_id]["instance"][ins_id]["score"] = score
            scoreList.append(score)

        max_score_id = instance_ids[scoreList.index(max(scoreList))]
        json_data[img_id]["image"]["1st_score"] = max_score_id
        high_scoreID = []
        s = 0
        img_name = json_data[img_id]["image"]["name"][:-4]

        for k in range(len(instance_ids)):
            ins_id = instance_ids[k]
            if ins_id == max_score_id or json_data[img_id]["instance"][ins_id]["score"] >= 1:
                high_scoreID.append(ins_id)
                # ins_name = '{}_{}'.format(img_name, ins_id)
                ins_file = '{}/instance/{}_{}.jpg'.format(_maskDir, img_name, ins_id)
                if s == 0:
                    mask_score = cv2.imread(ins_file, 0)
                else:
                    instance_mask = cv2.imread(ins_file, 0)
                    mask_score = cv2.add(mask_score, instance_mask)
                s += 1
        
        new_mask_dir = '{}/mask_top_score/{}.jpg'.format(_maskDir, img_name)
        cv2.imwrite(new_mask_dir, mask_score)
        json_data[img_id]["image"]["top_score"] = ", ".join(high_scoreID)
        if s != 1:
            print((i+1)/len(img_ids), " %  ############## ", img_name)
        if i+1 == len(img_ids):
            print("done")


    with open(jsonDir, 'w', encoding='utf-8') as make_file:
        json.dump(json_data, make_file, indent="\t")
    #print(json.dumps(json_data["785"], indent="\t"))

    pass


if __name__ == '__main__':
    # pass

    phase = "train"
    imgDir = './Dataset/COCO/{}2017_crop'.format(phase)
    maskDir = './Dataset/COCO/{}2017_crop_mask'.format(phase)
    jsonDir = './Dataset/COCO/{}2017_crop_mask/instance_info.json'.format(phase)
    make_score(imgDir, maskDir, jsonDir)
