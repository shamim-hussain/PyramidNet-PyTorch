
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models,optimizers,callbacks
from conv_invar import Conv2DR

num_rot=8

def conv3x3(x,filters, stride=1, kernel_size=(3,3),*args, **kwargs):
    "3x3 convolution with padding"
    x=Conv2DR(filters, kernel_size=kernel_size, strides=stride,
                         padding="SAME", use_bias=False,*args,**kwargs)(x)
    x=layers.Lambda(lambda v:tf.concat([v,v[...,:-1,:]],axis=-2))(x)
    x=layers.Conv3D(filters,(1,1,num_rot),use_bias=False)(x)
    return x

def conv1x1(x,filters, *args, **kwargs):
    "1x1 convolution with padding"
    return  layers.Dense(filters, use_bias=False,*args,**kwargs)(x)

def batchnorm(x,*args,**kwargs):
    return layers.BatchNormalization(epsilon=1e-05, momentum=0.1,*args,**kwargs)(x)

def activation(x,*args,**kwargs):
    return layers.Activation('relu',*args,**kwargs)(x)

class PadAdd(layers.Layer):
    def call(self, inputs):
        shortcut, out = inputs
        featuremap_size = shortcut.shape.as_list()[1:3]
        
        batch_size = tf.shape(out)[0]
        residual_channel = out.shape.as_list()[-1]
        shortcut_channel = shortcut.shape.as_list()[-1]

        if residual_channel != shortcut_channel:
            padding = tf.zeros([batch_size, 
                                featuremap_size[0], featuremap_size[1],num_rot,
                                residual_channel - shortcut_channel],
                                    dtype=tf.float32) 
            out = out + tf.concat((shortcut, padding), axis=-1)
        else:
            out = out + shortcut 
        
        return out

def basicblock(x,planes,stride,downsample=None):
    y=x
    
    x = batchnorm(x)
    x = conv3x3(x,planes, stride)        
    x = batchnorm(x)
    x = activation(x)
    x = conv3x3(x,planes)
    x = batchnorm(x)
    
    
    if downsample is not None:
        y = downsample(y)

    x = PadAdd()([y,x])   
    return x

basicblock.outchannel_ratio = 1

def bottleneck(x, planes, stride=1, downsample=None):
    y=x
    
    x = batchnorm(x)
    x = conv1x1(x,planes)

    x = batchnorm(x)
    x = activation(x)
    x = conv3x3(x,planes,stride)

    x = batchnorm(x)
    x = activation(x)
    x = conv1x1(x,planes * 4)

    x = batchnorm(x)
    
    
    if downsample is not None:
        y = downsample(y)
    
    x = PadAdd()([y,x])   
    return x

bottleneck.outchannel_ratio = 4



def pyramidal_make_layer(x, block, block_depth, 
                                stride, featuremap_dim, addrate):
    downsample = None
    if stride != 1: 
        downsample = layers.AveragePooling3D((2,2,1), strides = (2, 2, 1))
        
    featuremap_dim = featuremap_dim + addrate
    x= block(x, int(round(featuremap_dim)), stride, downsample)
    for i in range(1, block_depth):
        temp_featuremap_dim = featuremap_dim + addrate
        x=block(x,int(round(temp_featuremap_dim)), 1)
        featuremap_dim  = temp_featuremap_dim
    
    return x, featuremap_dim



def get_pyramidnet( dataset, depth, alpha, num_classes, bottleneck=False): 	
    if dataset.startswith('cifar'):
        inplanes = 16
        if bottleneck == True:
            n = int((depth - 2) / 9)
            block = bottleneck
        else:
            n = int((depth - 2) / 6)
            block = basicblock

        addrate = alpha / (3*n*1.0)
        
        inp = layers.Input(shape=(32,32,3))
        x = inp
        
        x=layers.Lambda(lambda v: tf.tile(v[...,None,:],[1,1,1,num_rot,1]))(x)
        
        x = conv3x3(x,inplanes,stride=1)
        x = batchnorm(x)

        featuremap_dim = inplanes
        x, featuremap_dim = pyramidal_make_layer(x,block, n, 1, featuremap_dim, addrate)
        x, featuremap_dim = pyramidal_make_layer(x,block, n, 2, featuremap_dim, addrate)
        x, featuremap_dim = pyramidal_make_layer(x,block, n, 2, featuremap_dim, addrate)

        
        x= batchnorm(x)
        x = activation(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(num_classes)(x)
        
        outp=x

    elif dataset == 'imagenet':
        blocks ={18: basicblock, 34: basicblock, 50: bottleneck, 101: bottleneck, 152: bottleneck, 200: bottleneck}
        layers_ ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}

        if layers_.get(depth) is None:
            if bottleneck == True:
                blocks[depth] = bottleneck
                temp_cfg = int((depth-2)/12)
            else:
                blocks[depth] = basicblock
                temp_cfg = int((depth-2)/8)

            layers_[depth]= [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
            print('=> the layer configuration for each stage is set to', layers_[depth])

        inplanes = 64            
        addrate = alpha / (sum(layers_[depth])*1.0)
        
        
        inp = layers.Input(shape=(32,32,3))
        x = inp
        
        x = conv3x3(x,inplanes,stride=2,kernel_size=7)
        # x = layers.Conv2D(inplanes, kernel_size=7,
        #                   stride=2, padding='same', use_bias=False)(x)
        
        x = batchnorm(x)
        x = activation(x)
        x = layers.MaxPool3D(pool_size=(3,3,1), strides=(2,2,1),
                             padding='same')(x)


        featuremap_dim = inplanes 
        x, featuremap_dim = pyramidal_make_layer(x, blocks[depth], layers_[depth][0], 1, featuremap_dim, addrate)
        x, featuremap_dim = pyramidal_make_layer(x, blocks[depth], layers_[depth][1], 2, featuremap_dim, addrate)
        x, featuremap_dim = pyramidal_make_layer(x, blocks[depth], layers_[depth][2], 2, featuremap_dim, addrate)
        x, featuremap_dim = pyramidal_make_layer(x, blocks[depth], layers_[depth][3], 2, featuremap_dim, addrate)

        x = batchnorm(x)
        x = activation(x)
        x = layers.GlobalAvergePooling3D()(x)
        x = layers.Dense(num_classes)(x)
        
        outp=x
    
    return models.Model(inp, outp)
        
        



