import cv2
import os
from copy import deepcopy

import inception
import darknet


class YOIO:
    def __init__(self, libdarknet_paths, inception_paths, train_imgs_path=None):
        self.yolo = darknet.YOLO(*libdarknet_paths)
        if train_imgs_path is not None:
            self.process_incept_train_data(train_imgs_path)
        else:
            self.incept_c = inception.InceptionClassifier(*inception_paths)

    def process_incept_train_data(self, train_imgs_dir_path):
        dir_data = os.walk(train_imgs_dir_path)
        dest_path = get_train_imgs_dest_path(dir_data[0])
        print('dest_path =', dest_path)
        curr_imgs = [dir_data[0] + '/' + name for name in dir_data[2]]

        if not os.path.exists(dest_path):
            os.mkdir(dest_path)

        for img_counter, img_path in enumerate(curr_imgs):
            print('img_path =', img_path)
            people_imgs = self.get_cropped_imgs(img_path)
            for crop_counter, img in enumerate(people_imgs):
                file_name = 'img_' + str(img_counter) + '_' + \
                    str(crop_counter) + '.' + img_path.split('.')[-1]
                print('file_name =', file_name)
                self.save_cropped_imgs([img], dest_path, file_name)

    def get_cropped_imgs(self, img_path, only_people=True):
        yolo_results = self.yolo.process_img(img_path)
        if only_people:
            yolo_results = self.only_people(found_objects=yolo_results)
        img = cv2.imread(img_path)
        return self.crop_imgs(yolo_results, img)

    def process_img(self, img_path):
        cropped_imgs = self.get_cropped_imgs(img_path, only_people=True)

        for crop_num, img in enumerate(cropped_imgs):
            cv2.imwrite('./test' + str(crop_num) + '.jpg', img)
            print('Crop numn =', crop_num)
            self.incept_c.classify_cropped_img(img)
            print('\n')

    def save_cropped_imgs(self, imgs, destination, file_name):
        for img in imgs:
            print('img dest =', destination + file_name)
            cv2.imwrite(destination + file_name, img)

    def only_people(self, found_objects):
        answer = []
        for obj in found_objects:
            if obj[0] == b'person':
                answer.append(deepcopy(obj))

        return answer

    def get_bbox_limits(self, center_x, center_y, width, height):
        delta_x = width // 2
        delta_y = height // 2
        min_i, max_i = center_y - delta_y, center_y + delta_y
        min_j, max_j = center_x - delta_x, center_x + delta_x
        min_i = min_i if min_i >= 0 else 0
        min_j = min_j if min_j >= 0 else 0
        max_i = max_i if max_i >= height else max_i
        max_j = max_j if max_j >= width else max_j
        return int(min_i), int(max_i), int(min_j), int(max_j)

    def crop_imgs(self, filtered_objs, img):
        cropped_imgs = []
        for obj in filtered_objs:
            i, I, j, J = self.get_bbox_limits(*obj[2])
            cropped_imgs.append(img[i:I, j:J, :])
        return cropped_imgs


def get_train_imgs_dest_path(current_path):
    label_name = current_path.split('/')[-1]
    return current_path + '/' + '../../preprocessed/' + label_name + '/'
