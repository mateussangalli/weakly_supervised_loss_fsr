from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Conv2DTranspose, DepthwiseConv2D, Dropout,
                          Input, Layer, LayerNormalization, MaxPool2D,
                          Multiply, UpSampling2D)
from keras.models import Model
from .unlabeled_training import SemiSupModel


class UNetBuilder:
    def __init__(self,
                 input_shape,
                 filters_start,
                 depth,
                 output_channels=3,
                 kernel_size=3,
                 normalization='none',
                 batch_norm_momentum=.99,
                 normalize_all=False,
                 activation='leaky_relu',
                 last_layer_activation='softmax',
                 drop_rate=0):
        self.input_shape = input_shape
        self.filters = [filters_start * 2 ** i for i in range(depth + 1)]
        self.depth = depth

        self.normalization = normalization.lower()
        self.use_normalization = self.normalization == 'batch' or self.normalization == 'layer'
        self.batch_norm_momentum = batch_norm_momentum
        self.normalize_all = normalize_all
        if not self.use_normalization:
            self.normalize_all = False

        self.activation = activation
        self.last_layer_activation = last_layer_activation
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.output_channels = output_channels

    def get_normalization(self, inputs):
        if self.normalization == 'batch':
            return BatchNormalization(momentum=self.batch_norm_momentum)(inputs)
        if self.normalization == 'layer':
            return LayerNormalization()(inputs)
        return inputs

    def conv_block(self, inputs, filters, strides=1):
        outputs = Conv2D(filters,
                         self.kernel_size,
                         strides=strides,
                         padding='same',
                         use_bias=not self.use_normalization)(inputs)
        outputs = self.get_normalization(outputs)
        outputs = Activation(self.activation)(outputs)

        outputs = Conv2D(filters,
                         self.kernel_size,
                         strides=1,
                         padding='same',
                         use_bias=not self.normalize_all)(outputs)
        if self.normalize_all:
            outputs = self.get_normalization(outputs)
        outputs = Activation(self.activation)(outputs)
        if self.drop_rate > 1e-5:
            outputs = Dropout(self.drop_rate)(outputs)
        return outputs

    def conv_upsample_block(self, feature_maps, skip, filters):
        feature_maps = UpSampling2D((2, 2))(feature_maps)
        feature_maps = Concatenate(-1)([skip, feature_maps])
        feature_maps = self.conv_block(feature_maps, filters)
        return feature_maps

    def build(self):
        inputs = Input(self.input_shape)
        feature_maps = inputs
        feature_maps = self.conv_block(feature_maps, self.filters[0])

        pool = [feature_maps]
        for i in range(1, self.depth + 1):
            feature_maps = self.conv_block(
                feature_maps, self.filters[i], strides=2)
            if self.drop_rate >= 0.001:
                feature_maps = Dropout(self.drop_rate)(feature_maps)
            pool.append(feature_maps)

        for i in range(1, self.depth + 1):
            skip = pool[-i - 1]
            feature_maps = self.conv_upsample_block(
                feature_maps, skip, self.filters[-i - 1])

        out = Conv2D(self.output_channels, kernel_size=3,
                     padding='same')(feature_maps)
        out = Activation(self.last_layer_activation, name='output_layer')(out)

        return Model(inputs, out)


class SemiSupUNetBuilder(UNetBuilder):
    def __init__(self,
                 input_shape,
                 filters_start,
                 depth,
                 alpha=0.,
                 output_channels=3,
                 kernel_size=3,
                 normalization='none',
                 batch_norm_momentum=.99,
                 normalize_all=False,
                 activation='leaky_relu',
                 last_layer_activation='softmax',
                 drop_rate=0):
        super().__init__(
            input_shape,
            filters_start,
            depth,
            output_channels,
            kernel_size,
            normalization,
            batch_norm_momentum,
            normalize_all,
            activation,
            last_layer_activation,
            drop_rate)
        self.alpha = alpha

    def build(self):
        inputs = Input(self.input_shape)
        feature_maps = inputs
        feature_maps = self.conv_block(feature_maps, self.filters[0])

        pool = [feature_maps]
        for i in range(1, self.depth + 1):
            feature_maps = self.conv_block(
                feature_maps, self.filters[i], strides=2)
            if self.drop_rate >= 0.001:
                feature_maps = Dropout(self.drop_rate)(feature_maps)
            pool.append(feature_maps)

        for i in range(1, self.depth + 1):
            skip = pool[-i - 1]
            feature_maps = self.conv_upsample_block(
                feature_maps, skip, self.filters[-i - 1])

        out = Conv2D(self.output_channels, kernel_size=3,
                     padding='same')(feature_maps)
        out = Activation(self.last_layer_activation, name='output_layer')(out)

        return SemiSupModel(inputs, out, alpha=self.alpha)
