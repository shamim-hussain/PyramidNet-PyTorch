
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models,optimizers,callbacks



def conv3x3(filters, stride=1, *args, **kwargs):
    "3x3 convolution with padding"
    return layers.Conv2D(filters, kernel_size=3, strides=stride,
                         padding="same", use_bias=False,*args,**kwargs)

def conv1x1(filters, *args, **kwargs):
    "1x1 convolution with padding"
    return  layers.Dense(filters, use_bias=False,*args,**kwargs)

def batchnorm(*args,**kwargs):
    return layers.BatchNormalization(epsilon=1e-05, momentum=0.1,*args,**kwargs)

def activation(*args,**kwargs):
    return layers.Activation('relu',*args,**kwargs)

class PadAdd(layers.Layer):
    @tf.function
    def call(self, inputs):
        shortcut, out = inputs
        featuremap_size = shortcut.shape.as_list()[1:3]
        
        batch_size = tf.shape(out)[0]
        residual_channel = out.shape.as_list()[-1]
        shortcut_channel = shortcut.shape.as_list()[-1]

        if residual_channel != shortcut_channel:
            padding = tf.zeros([batch_size, 
                                featuremap_size[0], featuremap_size[1],
                                residual_channel - shortcut_channel],
                                    dtype=tf.float32) 
            out = out + tf.concat((shortcut, padding), axis=-1)
        else:
            out = out + shortcut 
        
        return out

class BasicBlock(models.Model):
    outchannel_ratio = 1

    def __init__(self, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = batchnorm()
        self.conv1 = conv3x3(planes, stride)        
        self.bn2 = batchnorm()
        self.conv2 = conv3x3(planes)
        self.bn3 = batchnorm()
        self.relu = activation()
        
        self.downsample = downsample
        self.stride = stride
        
        self.pad = PadAdd()

    @tf.function
    def call(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        
        out = self.pad([shortcut,out])
        

        return out


class Bottleneck(models.Model):
    outchannel_ratio = 4

    def __init__(self, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = batchnorm()
        self.conv1 = conv1x1(planes)
        self.bn2 = batchnorm()
        self.conv2 = conv3x3(planes,stride)
        self.bn3 = batchnorm()
        self.conv3 = conv3x3(planes * Bottleneck.outchannel_ratio)
        self.bn4 = batchnorm()
        self.relu = activation()
        
        self.downsample = downsample
        self.stride = stride
        
        self.pad = PadAdd()
        
    @tf.function
    def call(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        
        out = self.pad([shortcut,out])

        return out



class PyramidNet(models.Model):
        
    def __init__(self, dataset, depth, alpha, num_classes, bottleneck=False):
        super(PyramidNet, self).__init__()   	
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.addrate = alpha / (3*n*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = conv3x3(self.input_featuremap_dim,stride=1,
                                 input_shape=(32,32,3))
            self.bn1 = batchnorm()

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(block, n)
            self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
            self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= batchnorm()
            self.relu_final = activation()
            self.avgpool = layers.GlobalAveragePooling2D()
            self.fc = layers.Dense(num_classes)

        elif dataset == 'imagenet':
            blocks ={18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
            layers_ ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

            if layers_.get(depth) is None:
                if bottleneck == True:
                    blocks[depth] = Bottleneck
                    temp_cfg = int((depth-2)/12)
                else:
                    blocks[depth] = BasicBlock
                    temp_cfg = int((depth-2)/8)

                layers_[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
                print('=> the layer configuration for each stage is set to', layers_[depth])

            self.inplanes = 64            
            self.addrate = alpha / (sum(layers_[depth])*1.0)

            self.input_featuremap_dim = self.inplanes
            self.conv1 = layers.Conv2D(self.input_featuremap_dim, kernel_size=7,
                                       stride=2, padding='same', use_bias=False,
                                       input_shape=(224,224,3))
            self.bn1 = batchnorm()
            self.relu = activation()
            self.maxpool = layers.MaxPool2D(pool_size=3, strides=2, padding='same')

            self.featuremap_dim = self.input_featuremap_dim 
            self.layer1 = self.pyramidal_make_layer(blocks[depth], layers_[depth][0])
            self.layer2 = self.pyramidal_make_layer(blocks[depth], layers_[depth][1], stride=2)
            self.layer3 = self.pyramidal_make_layer(blocks[depth], layers_[depth][2], stride=2)
            self.layer4 = self.pyramidal_make_layer(blocks[depth], layers_[depth][3], stride=2)

            self.final_featuremap_dim = self.input_featuremap_dim
            self.bn_final= batchnorm()
            self.relu_final = activation()
            self.avgpool = layers.GlobalAvergePooling2D()
            self.fc = layers.Dense(num_classes)
            
            
    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1: # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
            downsample = layers.AveragePooling2D((2,2), strides = (2, 2))

        layers_ = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers_.append(block(int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers_.append(block(int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return models.Sequential(layers_)
    
    @tf.function
    def call(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.bn_final(x)
            x = self.relu_final(x)
            x = self.avgpool(x)
            x = self.fc(x)
    
        return x

