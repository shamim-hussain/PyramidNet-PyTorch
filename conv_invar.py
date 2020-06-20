
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2


class Conv2DR(layers.Layer):
    RT_dict={}
    rotations=np.linspace(0,360,8,endpoint=False)
    
    def get_RT(self,ishape):
        if ishape in self.RT_dict:
            return self.RT_dict[ishape]
        
        fshape=np.prod(ishape)
        
        RT=np.zeros((len(self.rotations),fshape,fshape),dtype=np.float32)
        c=(ishape[0]/2-.5,ishape[1]/2-.5)
        eye=np.eye(fshape,dtype=np.float32).reshape(-1,*ishape)
        
        for j,theta in enumerate(self.rotations):
            M=cv2.getRotationMatrix2D(c,theta,1)
            for i in range(fshape):
                RT[j,i]=cv2.warpAffine(eye[i],M,ishape).reshape(-1)
        
        RT=RT.reshape(-1,*ishape,*ishape)
        RT=tf.constant(RT,tf.float32)
        
        self.RT_dict.update({ishape:RT})
        return RT
        
    
    def __init__(self,filters,kernel_size,
                 strides=(1,1),padding='VALID',activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 kernel_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        
        padding=padding.upper()
        
        if isinstance(kernel_size,int):
            kernel_size=(kernel_size,kernel_size)
        
        if isinstance(strides, int):
            strides=(strides,strides)
            
        self.filters=filters    
        self.kernel_size=kernel_size
        self.activation=activation
        self.padding=padding
        self.use_bias=use_bias
        self.strides=(1,strides[0],strides[1],1)
        
        self.kernel_initializer=kernel_initializer
        self.bias_initializer=bias_initializer
        self.kernel_regularizer=kernel_regularizer
        self.bias_regularizer=bias_regularizer
        self.kernel_constraint=kernel_constraint
        self.bias_constraint=bias_constraint
        
    def build(self,input_shape):
        self.RT=self.get_RT(self.kernel_size)
        
        self.kernel=self.add_weight('kernel',(self.kernel_size[0],
                                              self.kernel_size[1],
                                              input_shape[-1],
                                              self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        
        if self.use_bias:
            self.bias=self.add_weight('bias',(self.filters,),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)
        self.built=True
    
    def call(self,inputs):
        nrot=self.RT.shape[0]
        
        k=tf.tensordot(self.RT,self.kernel,[[1,2],[0,1]])
        
        outs=[]
        for i in range(nrot):
            o=tf.nn.conv2d(inputs[:,:,:,i,:], k[i,:,:,:,:],
                           self.strides, self.padding)
            outs.append(o)
        
        out=tf.stack(outs,axis=-2)
        
        if self.use_bias:
            b = self.bias
            out = out+b
        
        if not (self.activation is None):
            out = self.activation(out)
        
        return out
        
