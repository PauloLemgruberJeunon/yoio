from yoio import YOIO

if __name__ == '__main__':
    proj_path = '/home/paulojeunon/Desktop/Disciplinas/Sensores/t2/'

    darknet_path = proj_path + 'darknet/'
    libdarknet_paths = []
    libdarknet_paths.append(darknet_path + 'libdarknet.so')
    libdarknet_paths.append(darknet_path + 'cfg/yolov3.cfg')
    libdarknet_paths.append(darknet_path + 'yolov3.weights')
    libdarknet_paths.append(darknet_path + 'cfg/coco.data')

    inception_path = proj_path + 'inception/execution/trained/'
    inception_paths = []
    inception_paths.append(inception_path + 'output_graph.pb')
    inception_paths.append(inception_path + 'output_labels.txt')

    yoio = YOIO(libdarknet_paths, inception_paths,
                proj_path + 'inception/train_photos/raw/Paulo/')
    # yoio.process_img(proj_path + 'inception/demo_photos/Paulo/dom2.jpg')
