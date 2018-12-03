import cv2
from copy import deepcopy

import inception
import darknet


class YOIO:
    def __init__(self, libdarknet_paths, inception_paths):
        self.yolo = darknet.YOLO(*libdarknet_paths)
        self.incept_c = inception.InceptionClassifier(*inception_paths)

    def process_img(self, img_path):
        yolo_results = self.yolo.process_img(img_path)
        filtered_r = only_people(found_objects=yolo_results)
        img = cv2.imread(img_path)
        cropped_imgs = get_cropped_imgs(filtered_r, img)

        for crop_num, img in enumerate(cropped_imgs):
            cv2.imwrite('./test' + str(crop_num) + '.jpg', img)
            print('Crop numn =', crop_num)
            self.incept_c.classify_cropped_img(img)
            print('\n')


def only_people(found_objects):
    answer = []
    for obj in found_objects:
        if obj[0] == b'person':
            answer.append(deepcopy(obj))

    return answer


def get_bbox_limits(center_x, center_y, width, height):
    delta_x = width // 2
    delta_y = height // 2
    min_i, max_i = center_y - delta_y, center_y + delta_y
    min_j, max_j = center_x - delta_x, center_x + delta_x
    min_i = min_i if min_i >= 0 else 0
    min_j = min_j if min_j >= 0 else 0
    max_i = max_i if max_i >= 0 else 0
    max_j = max_j if max_j >= 0 else 0
    return int(min_i), int(max_i), int(min_j), int(max_j)


def get_cropped_imgs(filtered_objs, img):
    cropped_imgs = []
    for obj in filtered_objs:
        i, I, j, J = get_bbox_limits(*obj[2])
        cropped_imgs.append(img[i:I, j:J, :])
    return cropped_imgs
