"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function

import json
import os

import numpy as np
# matplotlib.use('Agg')
import scipy.misc
import imageio
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import concurrent.futures
import time

labels=[]
def split_to_coco_creator(input_instance_array, labels):

    labelid_matrix_name = []

    t0=time.time()
    label_image_info = np.array(input_instance_array / 256, dtype=np.uint8)

    instance_image_info = np.array(input_instance_array % 256, dtype=np.uint8)
    t1=time.time()

    unique_label_info = np.unique(label_image_info)
    unique_instance_info = np.unique(instance_image_info)
    t2=time.time()
    #print("a",t2-t1,t1-t0)
    for label_id, label in enumerate(labels):
        t00=time.time()
        if (label_id in (unique_label_info)) and (label["instances"] == True):

            each_label_array = np.zeros((input_instance_array.shape[0], input_instance_array.shape[1]),
                                        dtype=np.uint8)

            each_label_array[label_image_info == label_id] = 255
            t01=time.time()
            #print("b",t01-t00)
            for instance_id in range(256):
                if (instance_id in unique_instance_info):
                    each_instance_array = np.zeros(
                        (input_instance_array.shape[
                         0], input_instance_array.shape[1]),
                        dtype=np.uint8)

                    each_instance_array[
                        instance_image_info == instance_id] = 255

                    final_instance_array = np.bitwise_and(
                        each_instance_array, each_label_array)

                    if np.unique(final_instance_array).size == 2:
                        labelid_matrix_name.append(
                            {"label_id": label_id, "instance_id": instance_id,
                             "label_name": label["readable"],
                             "image": final_instance_array})
    #print("c",time.time()-t0)
    # each_id_array [(input_instance_array % 256) == instance_id] = 1
    # labelid_matrix_name.append ( (label_id , instance_id, label [ "readable"
    # ] , each_id_array) )

    return labelid_matrix_name


def split_dir(dir_name):
    print ("Spliting {}".format(dir_name))

    dir_path = "{}/instances".format(dir_name)
    files = os.listdir(dir_path)
    # read in config file
    with open('config.json') as config_file:
        config = json.load(config_file)

    labels = config['labels']
    
    def process(file_name):
        print(file_name)
        file_name = file_name[:-4]
        instance_path = "{}/instances/{}.png".format(dir_name,file_name)
        instance_image = Image.open(instance_path)
        instance_array = np.array(instance_image, dtype=np.uint16)
        image_label_instance_infomatrix = split_to_coco_creator(
            instance_array, labels)

        for item in image_label_instance_infomatrix:
            path = "{}_{}_{}.png".format(
                file_name, item["label_name"].replace(" ", "_"), item["instance_id"])
            imageio.imsave("{}/annotations/{}".format(dir_name,path), item["image"])
        del image_label_instance_infomatrix
        del instance_image
        del instance_array
        return 0
    
    def run(f, my_iter):
        with concurrent.futures.ThreadPoolExecutor(2) as executor:
            results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    
    run(process, files)

        

if __name__ == '__main__':
    split_dir("training")
