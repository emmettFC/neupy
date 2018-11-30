import tensorflow as tf

from neupy.utils import as_tuple
from neupy.layers import *
from neupy import architectures, plots


class ResizeBilinear(BaseLayer):
    def __init__(self, new_shape, *args, **kwargs):
        self.new_shape = new_shape
        super(ResizeBilinear, self).__init__(*args, **kwargs)

    @property
    def output_shape(self):
        if self.input_shape:
            return as_tuple(self.new_shape, self.input_shape[-1])

    def output(self, input_value):
        return tf.image.resize_bilinear(input_value, self.new_shape)


resnet50 = architectures.resnet50((224, 224, 3))
layer_before_global_pool = resnet50.layers[-3]
resnet50 = resnet50.end(layer_before_global_pool)

deeplab = join(
    # Part of the ResNet-50 network
    resnet50,

    # Atrous Spatial Pyramid Pooling
    [[
        Convolution((1, 1, 256), bias=None, padding='same'),
        BatchNorm() > Relu(),
    ], [
        Convolution((3, 3, 256), bias=None, padding='same'),
        BatchNorm() > Relu(),
    ], [
        Convolution((3, 3, 256), bias=None, padding='same'),
        BatchNorm() > Relu(),
    ], [
        Convolution((3, 3, 256), bias=None, padding='same'),
        BatchNorm() > Relu(),
    ], [
        GlobalPooling('avg') > Reshape((1, 1, -1)),

        Convolution((1, 1, 256), bias=None, padding='same'),
        BatchNorm() > Relu(),

        ResizeBilinear(resnet50.output_shape[:2]),
    ]],
    Concatenate(),

    Convolution((1, 1, 256), bias=None, padding='same'),
    BatchNorm() > Relu(),

    # Convert to the classification maps
    Convolution((1, 1, 21), padding='same'),
    ResizeBilinear(resnet50.input_shape[:2]),
)
# plots.layer_structure(deeplab)
