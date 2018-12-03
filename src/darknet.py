from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ctypes import POINTER, c_float, c_int, c_char_p, c_void_p, pointer
from ctypes import Structure
from ctypes import CDLL, RTLD_GLOBAL
import random


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class YOLO:
    def __init__(self, libdarknet_path, cfg_path, weights_path, meta_path):
        self.configure(libdarknet_path)
        self.net = self.load_net(bytes(cfg_path, 'ascii'), bytes(weights_path,
                                                                 'ascii'), 0)
        self.meta = self.load_meta(bytes(meta_path, 'ascii'))

    def process_img(self, img_path):
        img_path = bytes(img_path, 'ascii')
        result = self.detect(img_path)
        return result

    def classify(self, im):
        out = self.predict_image(self.net, im)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh,
                                      None, 0, pnum)
        num = pnum[0]
        if (nms):
            self.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i],
                                (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def configure(self, so_path):
        so_path = bytes(so_path, 'ascii')
        self.lib = CDLL(so_path, RTLD_GLOBAL)
        self.predict = self.lib.network_predict
        self.set_gpu = self.lib.cuda_set_device
        self.make_image = self.lib.make_image
        self.get_network_boxes = self.lib.get_network_boxes
        self.make_network_boxes = self.lib.make_network_boxes
        self.free_detections = self.lib.free_detections
        self.free_ptrs = self.lib.free_ptrs
        self.network_predict = self.lib.network_predict
        self.reset_rnn = self.lib.reset_rnn
        self.load_net = self.lib.load_network
        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_sort = self.lib.do_nms_sort
        self.free_image = self.lib.free_image
        self.letterbox_image = self.lib.letterbox_image
        self.load_meta = self.lib.get_metadata
        self.load_image = self.lib.load_image_color
        self.rgbgr_image = self.lib.rgbgr_image
        self.predict_image = self.lib.network_predict_image

        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu.argtypes = [c_int]

        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int,
                                           c_float, c_float, POINTER(c_int), c_int,
                                           POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn.argtypes = [c_void_p]

        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image.argtypes = [IMAGE]

        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)


if __name__ == "__main__":
    libdarknet_path = "/home/paulojeunon/Desktop/Disciplinas/Sensores/t2/darknet/libdarknet.so"
    cfg_path = '/home/paulojeunon/Desktop/Disciplinas/Sensores/t2/darknet/cfg/yolov3.cfg'
    weights_path = '/home/paulojeunon/Desktop/Disciplinas/Sensores/t2/darknet/yolov3.weights'
    meta_path = '/home/paulojeunon/Desktop/Disciplinas/Sensores/t2/darknet/cfg/coco.data'
    yolo = YOLO(libdarknet_path, cfg_path, weights_path, meta_path)
    print(yolo.process_img(img_path='/home/paulojeunon/t01_photos/final/test_photos/dom8.jpg'))
