import tensorflow as tf
import numpy as np


class InceptionClassifier:
    def __init__(self, model_file, label_file):
        self.model_file = model_file
        self.label_file = label_file
        self.input_layer = 'Placeholder'
        self.output_layer = 'final_result'
        self.graph = self.load_graph()

        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

    def classify_cropped_img(self, cropped_img):
        t = self.read_tensor_from_image_file(cropped_img)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels()
        for i in top_k:
            print(labels[i], results[i])

    def load_labels(self):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(self.label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image_file(self, tensor_img, input_height=299,
                                    input_width=299):
        image_reader = tensor_img
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [0]), [255])
        sess = tf.Session()
        result = sess.run(normalized)
        return result

    def load_graph(self):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(self.model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph
