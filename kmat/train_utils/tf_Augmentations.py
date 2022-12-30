# -*- coding: utf-8 -*-
"""
@author: k_mat

"""
#import random
import math

import numpy as np
import tensorflow as tf

FLOAT_TYPE = tf.float32
INT_TYPE = tf.int32


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
            
    def __call__(self, inputs):#, targets):
        inputs["rectangles"] = tf.reshape(inputs["rectangles"], [-1,2])
        inputs["locations"] = tf.reshape(inputs["locations"], [-1,2])

        for i, transform in enumerate(self.transforms):
            
            if tf.random.uniform(()) > transform.p:
                continue
            transform.reset_params()
            
            inputs["rgb"] = transform(inputs["rgb"], "rgb")
            inputs["rectangles"] = transform(inputs["rectangles"], "quadrangles")
            inputs["locations"] = transform(inputs["locations"], "locations")
            
        inputs["rectangles"] = tf.reshape(inputs["rectangles"], [-1,4,2])
        inputs["locations"] = tf.reshape(inputs["locations"], [-1,1,2])

        return inputs

class Oneof():
    
    def __init__(self, transforms, probs):
        """
        
        probs : probability for each augmentations. set like [0.5, 0.2] (in this case, no augmentation will be conducted rest of 30%)
                total probabilities must be smaller than 1.0.
                probabilities for each augmentation will be neglected.
        """
        if len(transforms)!=len(probs):
            raise Exception("Set probabilities for One of Augmentation. len(transforms) must be the same as len(probs).")
        if np.sum(probs)>1.0:
            raise Exception("Sum of probabilities for One of Augmentation shall not be bigger than 1.0")
        for transform in transforms:
            if transform.p <1.0:
                print("probabilities for each augmentation will be neglected in One of Aug.")
                transform.p = 1.0
                
        self.num_transforms = len(transforms)
        self.transforms = transforms
        self.total_probs = [np.sum(probs[:i]) for i in range(1,1+self.num_transforms)]# + [1.0]
        self.p = 1.0
        
    def select(self, rand):
        num = -1
        for i, prob in enumerate(self.total_probs):
            if prob < rand:
                num = i
        return num
    
    def reset_params(self):
        self.rand = tf.random.uniform(())#random value for choosing which transformer to use.
        for trans in self.transforms:
            trans.reset_params()
        
    
    def __call__(self, img, key):
        out_img = img# in case of no transform
        done = False
        for trans, prob in zip(self.transforms, self.total_probs):
            if self.rand<=prob and done==False:
                out_img = trans(img, key)
                done = True
        return out_img
    

class VerticalFlip():
    """Flip the input vertically around the x-axis.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, inputs, key):

        if key in ["quadrangles"]:
            inputs = self.apply_to_point(inputs)
        else:
            inputs = self.apply(inputs)
        return inputs

    #@tf.function     
    def reset_params(self):
        pass 
    
    def get_params(self):
        return {}   
    
    def apply(self, img):
        return tf.image.flip_up_down(img)
    
    def apply_to_point(self, inputs):#shape: [point_num, 2(x,y)]
        #return tf.constant([[0,1]], tf.float32) + (inputs * tf.constant([[1,-1]], tf.float32))
        return tf.cast(tf.stack([[0,self.height-1]]), tf.float32) + (inputs * tf.constant([[1,-1]], tf.float32))


class HorizontalFlip():
    """
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, inputs, key):
        if key in ["quadrangles"]:
            inputs = self.apply_to_point(inputs)
        elif key in ["locations"]:
            inputs = self.apply_to_location(inputs)            
        else:
            inputs = self.apply(inputs)
        return inputs

    #@tf.function     
    def reset_params(self):
        self.height = None
        self.width = None
    
    def get_params(self):
        return {}   
    
    def apply(self, img):
        self.height, self.width = tf.unstack(tf.shape(img))[:2]
        return tf.image.flip_left_right(img)

    def apply_to_point(self, inputs):#shape: [point_num, 2(x,y)]
        #return tf.constant([[1,0]], tf.float32) + (inputs * tf.constant([[-1,1]], tf.float32))
        return tf.cast(tf.stack([[self.width-1,0]]), tf.float32) + (inputs * tf.constant([[-1,1]], tf.float32))
    
    def apply_to_location(self, inputs):
        return tf.constant([[120,0]], tf.float32) + inputs * tf.constant([[-1,1]], tf.float32)


class Resize():
    """
    """
    def __init__(self, height, width, target_height, target_width):
        self.p = 1.0
        self.height = height
        self.width = width
        self.target_height = target_height
        self.target_width = target_width
        
    def __call__(self, inputs, key):
            
        if key in ["rgb"]:
            if self.height is not None:
                inputs = self.apply(inputs, self.height, self.width, "nearest")#bilinear
        elif key in ["quadrangles"]:
            if self.height is not None:
                inputs = self.apply_to_point(inputs)
        elif key in ["locations"]:
            inputs = inputs
        else:
            if self.target_height is not None:
                inputs = self.apply(inputs, self.target_height, self.target_width, "nearest")
        
        return inputs

    def reset_params(self):
        pass
    
    def get_params(self):
        return {}   
    
    def apply(self, img, height, width, interpolation):
        original_shape = tf.shape(img)
        self.original_height, self.original_width = original_shape[0], original_shape[1]
        img = img[tf.newaxis, ...]
        return tf.image.resize(img, (height, width), method=interpolation)[0,:,:,:]
    
    def apply_to_point(self, inputs, dev_h=0, dev_w=0):#既にノーマライズしていれば不要
        h_rate = (self.height+dev_h)/self.original_height
        w_rate = (self.width+dev_w)/self.original_width
        return inputs * tf.cast(tf.stack([[w_rate,h_rate]]), tf.float32)
         
class Crop():
    """
    """
    def __init__(self, p=1.0, min_height=0.8, min_width=0.8, max_height=None, max_width=None):
        self.p = p
        self.min_height = min_height
        self.min_width = min_width
        self.max_width = max_width if max_width is not None else min_width
        self.max_height = max_height if max_height is not None else min_height

    def __call__(self, data, key):
        if key in ["quadrangles"]:
            data = self.apply_to_point(data)
        elif key in ["locations"]:
            data = data            
        else:
            data = self.apply(data)
        return data
    
    def reset_params(self):
        self.top = None
        self.left = None
        self.bottom = None
        self.right = None
    
    def get_params(self):
        return {"top":self.top, "left":self.left, "bottom":self.bottom, "right":self.right}   
    
    def apply(self, img):
        if self.top is None:
            img_height, img_width, _ = tf.unstack(tf.shape(img))
            height_ratio = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.max_height+1e-7, dtype=tf.float32)#random.randint(self.min_height, self.max_height)
            width_ratio = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.max_width+1e-7, dtype=tf.float32)#random.randint(self.min_width, self.max_width)
            crop_height = tf.cast(tf.cast(img_height, tf.float32) * height_ratio, tf.int32)
            crop_width = tf.cast(tf.cast(img_width, tf.float32) * width_ratio, tf.int32)
            self.top = tf.random.uniform(shape=[], minval=0, maxval=img_height-crop_height+1, dtype=tf.int32)#random.randint(0, img_height-crop_height)
            self.left = tf.random.uniform(shape=[], minval=0, maxval=img_width-crop_width+1, dtype=tf.int32)#random.randint(0, img_width-crop_width)
            self.bottom = self.top + crop_height
            self.right = self.left + crop_width
        return img[self.top:self.bottom, self.left:self.right, :]
    
    def apply_to_point(self, inputs):#shape: [point_num, 2(x,y)]
        return inputs - tf.cast([[self.left, self.top]], tf.float32)

    
class Center_Crop(Crop):
    
    def apply(self, img):
        if self.top is None:
            img_height, img_width, _ = tf.unstack(tf.shape(img))
            height_ratio = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.max_height+1e-7, dtype=tf.float32)#random.randint(self.min_height, self.max_height)
            width_ratio = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.max_width+1e-7, dtype=tf.float32)#random.randint(self.min_width, self.max_width)
            crop_height = tf.cast(tf.cast(img_height, tf.float32) * height_ratio, tf.int32)
            crop_width = tf.cast(tf.cast(img_width, tf.float32) * width_ratio, tf.int32)
   
            self.top = (img_height-crop_height)//2#tf.random.uniform(shape=[], minval=0, maxval=img_height-crop_height+1, dtype=tf.int32)#random.randint(0, img_height-crop_height)
            self.left = (img_width-crop_width)//2#tf.random.uniform(shape=[], minval=0, maxval=img_width-crop_width+1, dtype=tf.int32)#random.randint(0, img_width-crop_width)
            self.bottom = self.top + crop_height
            self.right = self.left + crop_width
        return img[self.top:self.bottom, self.left:self.right, :]

class GaussianNoise():
    """
    assuming uint max255
    """
    def __init__(self, p=0.3, min_var=10, max_var=50):#, min_multiply=0.8, max_multiply=1.2):
        self.p = p
        self.min_var, self.max_var = min_var, max_var
        
    def __call__(self, img, key):
        if key in ["rgb"]:
            img = self.apply(img)
        return img

    def reset_params(self):
        self.stddev = tf.random.uniform(shape=[], minval=self.min_var, maxval=self.max_var, dtype=FLOAT_TYPE)
        

    def get_params(self):
        return {"offset": self.offset, "multiply": self.multiply}
    
    def apply(self, img):
        noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=self.stddev, dtype=tf.float32)
        img = tf.clip_by_value(img+noise, 0, 255)
        return img
    
class BrightnessContrast():
    """
    assuming uint max255
    """
    def __init__(self, p=1.0, min_offset=-20, max_offset=20, min_multiply=0.8, max_multiply=1.2):
        self.p = p
        self.min_offset, self.max_offset = min_offset, max_offset
        self.min_multiply, self.max_multiply = min_multiply, max_multiply
        
    def __call__(self, img, key):
        if key in ["rgb"]:
            img = self.apply(img)
        return img

    def reset_params(self):
        self.offset = tf.random.uniform(shape=[], minval=self.min_offset, maxval=self.max_offset, dtype=FLOAT_TYPE)
        self.multiply = tf.random.uniform(shape=[], minval=self.min_multiply, maxval=self.max_multiply, dtype=FLOAT_TYPE)

    def get_params(self):
        return {"offset": self.offset, "multiply": self.multiply}
    
    def apply(self, img):
        img = tf.image.adjust_contrast(img, self.multiply)
        img = tf.image.adjust_brightness(img, self.offset)
        img = tf.clip_by_value(img, 0, 255)
        return img

class BoxShift():
    """
    """
    def __init__(self, p=0.5, min_offset=-0.1, max_offset=0.1):
        self.p = p
        self.min_offset, self.max_offset = min_offset, max_offset
        
    def __call__(self, inputs, key):
        if key in ["quadrangles"]:
            inputs = self.apply(inputs)
        return inputs

    def reset_params(self):
        pass
        
    def get_params(self):
        return {"offset": self.offset}
    
    def apply(self, inputs):
        inputs = tf.reshape(inputs, [-1,4,2])
        num_box, _, _ = tf.unstack(tf.shape(inputs))
        left_top = tf.reduce_min(inputs, axis=1, keepdims=True)
        right_bottom = tf.reduce_max(inputs, axis=1, keepdims=True)
        width_height = right_bottom - left_top
        shifts = tf.random.uniform((num_box,2,2), self.min_offset*width_height, self.max_offset*width_height)        
        p1 = inputs[:,0:1,:] + shifts[:,0:1,:] * tf.constant([[[1.0, 1.0]]]) + shifts[:,1:2,:] * tf.constant([[[0.0, 0.0]]])
        p2 = inputs[:,1:2,:] + shifts[:,0:1,:] * tf.constant([[[0.0, 1.0]]]) + shifts[:,1:2,:] * tf.constant([[[1.0, 0.0]]])
        p3 = inputs[:,2:3,:] + shifts[:,0:1,:] * tf.constant([[[0.0, 0.0]]]) + shifts[:,1:2,:] * tf.constant([[[1.0, 1.0]]])
        p4 = inputs[:,3:4,:] + shifts[:,0:1,:] * tf.constant([[[1.0, 0.0]]]) + shifts[:,1:2,:] * tf.constant([[[0.0, 1.0]]])
        outputs = tf.reshape(tf.concat([p1,p2,p3,p4], axis=1), [-1,2])
        return outputs


class HueShift():
    """
    assuming uint max255
    """
    def __init__(self, p=1.0, min_offset=-0.20, max_offset=0.20):
        self.p = p
        self.min_offset, self.max_offset = min_offset, max_offset
        
    def __call__(self, img, key):
        if key in ["rgb"]:
            img = self.apply(img)
        return img

    def reset_params(self):
        self.offset = tf.random.uniform(shape=[], minval=self.min_offset, maxval=self.max_offset, dtype=FLOAT_TYPE)

    def get_params(self):
        return {"offset": self.offset}
    
    def apply(self, img):
        return tf.image.adjust_hue(img, self.offset)

class ToGlay():
    """
    mean is not the best. should be changed
    """
    def __init__(self, p=0.2):
        self.p = p
        
    def __call__(self, inputs, key):

        if key in ["rgb"]:
            inputs = self.apply(inputs)
        return inputs

    def reset_params(self):
        pass 
    
    def get_params(self):
        return {}   
    
    def apply(self, img):
        return tf.tile(tf.image.rgb_to_grayscale(img), [1,1,3])


class PertialBrightnessContrast():
    
    def __init__(self, p=0.5, 
                 max_holes=8, max_height=8, max_width=8,
                 min_holes=None, min_height=None, min_width=None,
                 min_offset=-20, max_offset=20, min_multiply=0.8, max_multiply=1.2,
                 rect_ratio=0.4, 
                 #affect_depth=False,
                 #depth_drop_ratio=0.5, depth_outlier_ratio=0.2, 
                 #max_outlier_ratio=0.5
                 ):
        
        
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.min_offset, self.max_offset = min_offset, max_offset
        self.min_multiply, self.max_multiply = min_multiply, max_multiply
        self.rect_ratio = rect_ratio
        #self.affect_depth = affect_depth
        #self.depth_drop_ratio = depth_drop_ratio
        #self.depth_outlier_ratio = depth_outlier_ratio
        #self.max_outlier_ratio = max_outlier_ratio

        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))
        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))
        #if (self.depth_drop_ratio + self.depth_outlier_ratio)>1.0:
        #    raise ValueError("total ratio (depth_drop_ratio + depth_outlier_ratio) must be smaller than 1.0. Got: {}".format([depth_drop_ratio, depth_outlier_ratio]))


    def __call__(self, inputs, key):
        if key in ["rgb"]:
            inputs = self.apply(inputs)
        return inputs
    
    def reset_params(self):
        self.holes = None
        self.brightness = None
        self.offset = tf.random.uniform(shape=[], minval=self.min_offset, maxval=self.max_offset, dtype=FLOAT_TYPE)
        self.multiply = tf.random.uniform(shape=[], minval=self.min_multiply, maxval=self.max_multiply, dtype=FLOAT_TYPE)
    
    def get_params(self):
        return {"holes":self.holes}   
    
    def apply(self, img):
        img_shape = tf.shape(img)
        img_height, img_width = img_shape[0], img_shape[1]
        if self.holes is None:
            
            self.holes = tf.ones_like(img)
            num_holes = tf.random.uniform(shape=[], minval=self.min_holes, maxval=self.max_holes+1, dtype=INT_TYPE)
            def cond(i, x):
                return i < num_holes
            def body(i, x):
                #sigma = 10.0#tf.random.uniform(shape=[], minval=0.1, maxval=2.0, dtype=tf.float32)
                sigma = tf.random.uniform(shape=[], minval=5.0, maxval=20.0, dtype=tf.float32)
                #max_range = 15#tf.random.uniform(shape=[], minval=0.1, maxval=2.0, dtype=tf.float32)
        
                hole_height = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.max_height+1, dtype=INT_TYPE)
                hole_width = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.max_width+1, dtype=INT_TYPE)
    
                top = tf.random.uniform(shape=[], minval=0, maxval=img_height-hole_height+1, dtype=INT_TYPE)
                left = tf.random.uniform(shape=[], minval=0, maxval=img_width-hole_width+1, dtype=INT_TYPE)
                bottom = top + hole_height
                right = left + hole_width
                if tf.random.uniform(()) < self.rect_ratio:# rectangle shadow
                    h_dist = tf.maximum(top - tf.range(img_height), 0) + tf.maximum(tf.range(img_height) - bottom, 0)
                    w_dist = tf.maximum(left - tf.range(img_width), 0) + tf.maximum(tf.range(img_width) - right, 0)
                    h_dist = tf.cast(h_dist, tf.float32)[:,tf.newaxis]
                    w_dist = tf.cast(w_dist, tf.float32)[tf.newaxis,:]
                    gauss_dist = tf.math.exp(-((h_dist**2 + w_dist**2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
                
                else:#ellipse shadow
                    h_dist = tf.cast((top+bottom)//2 - tf.range(img_height), tf.float32)[:,tf.newaxis]
                    w_dist = tf.cast((left+right)//2 - tf.range(img_width), tf.float32)[tf.newaxis, :]
                    r_h = tf.cast((bottom-top)//2, tf.float32)
                    r_w = tf.cast((right-left)//2, tf.float32)
                    h_dist_rate = h_dist / r_h
                    w_dist_rate = w_dist / r_w
                    dist_rate = tf.maximum(tf.math.sqrt(h_dist_rate**2 + w_dist_rate**2)-1.0, 0.0)
                    
                    rate_back = ((r_h * h_dist)**2 + (r_w * w_dist)**2)/(h_dist**2 + w_dist**2 +1e-6)
                    gauss_dist = tf.math.exp(-(((dist_rate**2) * rate_back) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
               
                x = x * (1.0 - gauss_dist[..., tf.newaxis])#0 is light
                return i + 1, x
            _, self.holes = tf.while_loop(cond, body, (0, self.holes), shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, maximum_iterations=None, name=None)
            

        img = img + img * ((self.multiply-1.0)*(1.0-self.holes))
        img = img + self.offset * (1-self.holes)# * self.fill_value
        img = tf.clip_by_value(img, 0, 255)
        self.brightness = tf.reduce_mean(img, axis=-1, keepdims=True)
        return img


class Shadow():
    
    def __init__(self, p=0.5, 
                 max_holes=8, max_height=8, max_width=8,
                 min_holes=None, min_height=None, min_width=None,
                 min_strength=0.2, max_strength=0.8, shadow_color=0.0, rect_ratio=0.5):
        
        
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.shadow_color = shadow_color
        self.rect_ratio = rect_ratio
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))
        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))


    def __call__(self, inputs, key):
        if key in ["rgb"]:
            inputs = self.apply(inputs)
        return inputs
    
    def reset_params(self):
        self.holes = None
        self.shadow_strength = tf.random.uniform(shape=[], minval=self.min_strength, maxval=self.max_strength, dtype=FLOAT_TYPE)
    
    def get_params(self):
        return {"holes":self.holes}   
    
    def apply(self, img):
        img_shape = tf.shape(img)
        img_height, img_width = img_shape[0], img_shape[1]
        if self.holes is None:
            
            self.holes = tf.ones_like(img)
            num_holes = tf.random.uniform(shape=[], minval=self.min_holes, maxval=self.max_holes+1, dtype=INT_TYPE)
            def cond(i, x):
                return i < num_holes
            def body(i, x):
                # small sigma -> sharp shadow.  large sigma -> blurred shadow
                sigma = tf.random.uniform(shape=[], minval=5.0, maxval=20.0, dtype=tf.float32)
        
                hole_height = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.max_height+1, dtype=INT_TYPE)
                hole_width = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.max_width+1, dtype=INT_TYPE)
    
                top = tf.random.uniform(shape=[], minval=0, maxval=img_height-hole_height+1, dtype=INT_TYPE)
                left = tf.random.uniform(shape=[], minval=0, maxval=img_width-hole_width+1, dtype=INT_TYPE)
                bottom = top + hole_height
                right = left + hole_width
                
                
                if tf.random.uniform(()) < self.rect_ratio:# rectangle shadow
                    h_dist = tf.maximum(top - tf.range(img_height), 0) + tf.maximum(tf.range(img_height) - bottom, 0)
                    w_dist = tf.maximum(left - tf.range(img_width), 0) + tf.maximum(tf.range(img_width) - right, 0)
                    h_dist = tf.cast(h_dist, tf.float32)[:,tf.newaxis]
                    w_dist = tf.cast(w_dist, tf.float32)[tf.newaxis,:]
                    gauss_dist = tf.math.exp(-((h_dist**2 + w_dist**2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
                
                else:#ellipse shadow
                    h_dist = tf.cast((top+bottom)//2 - tf.range(img_height), tf.float32)[:,tf.newaxis]
                    w_dist = tf.cast((left+right)//2 - tf.range(img_width), tf.float32)[tf.newaxis, :]
                    r_h = tf.cast((bottom-top)//2, tf.float32)
                    r_w = tf.cast((right-left)//2, tf.float32)
                    h_dist_rate = h_dist / r_h
                    w_dist_rate = w_dist / r_w
                    dist_rate = tf.maximum(tf.math.sqrt(h_dist_rate**2 + w_dist_rate**2)-1.0, 0.0)
                    
                    rate_back = ((r_h * h_dist)**2 + (r_w * w_dist)**2)/(h_dist**2 + w_dist**2 +1e-6)
                    gauss_dist = tf.math.exp(-(((dist_rate**2) * rate_back) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))

                x = x * (1.0 - gauss_dist[..., tf.newaxis])#0 is shadow(or light)
                return i + 1, x
            _, self.holes = tf.while_loop(cond, body, (0, self.holes), shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, maximum_iterations=None, name=None)
            
        shadow_mask = self.shadow_strength * (1.0 - self.holes)
        img = img * (1.0 - shadow_mask) + shadow_mask * self.shadow_color
        return img


class Blur():
    """
    """
    def __init__(self, p=0.5):
        self.p = p
        #self.blur = self._make_kernel(3, 1, 3)

        
    def __call__(self, inputs, key):

        if key in ["rgb"]:
            inputs = self.apply(inputs)
        return inputs

    def _make_kernel(self, kernel_size, sigma, n_channels):
        x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=tf.float32)
        g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, tf.float32), 2))))
        g_norm2d = tf.pow(tf.reduce_sum(g), 2)
        g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
        g_kernel = tf.expand_dims(g_kernel, axis=-1)
        return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

    def reset_params(self):
        pass 
    
    def get_params(self):
        return {}   
    
    def apply(self, img):
        _, _, num_channnel = tf.unstack(tf.shape(img))
        sigma = tf.random.uniform(shape=[], minval=0.1, maxval=2.0, dtype=tf.float32)
        blur = self._make_kernel(3, sigma, num_channnel)
        img = tf.nn.depthwise_conv2d(img[tf.newaxis,...], blur, [1,1,1,1], 'SAME')
        return img[0,...]
    

class CoarseDropout():
    
    def __init__(self, p=0.5, 
                 max_holes=8, max_height=8, max_width=8,
                 min_holes=None, min_height=None, min_width=None,
                 fill_value=0, 
                 ):
        self.p = p
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError("Invalid combination of min_holes and max_holes. Got: {}".format([min_holes, max_holes]))
        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format([min_height, max_height])
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError("Invalid combination of min_width and max_width. Got: {}".format([min_width, max_width]))

    def __call__(self, img, key):
        if key in ["rgb"]:
            img = self.apply(img)
        return img
    
    def reset_params(self):
        self.holes = None
    
    def get_params(self):
        return {"holes":self.holes}   
    
    def apply(self, img):
        img_shape = tf.shape(img)
        img_height, img_width = img_shape[0], img_shape[1]
        if self.holes is None:
            
            self.holes = tf.ones_like(img)
            num_holes = tf.random.uniform(shape=[], minval=self.min_holes, maxval=self.max_holes+1, dtype=INT_TYPE)
            def cond(i, x):
                return i < num_holes
            def body(i, x):
                hole_height = tf.random.uniform(shape=[], minval=self.min_height, maxval=self.max_height+1, dtype=INT_TYPE)
                hole_width = tf.random.uniform(shape=[], minval=self.min_width, maxval=self.max_width+1, dtype=INT_TYPE)
    
                top = tf.random.uniform(shape=[], minval=0, maxval=img_height-hole_height+1, dtype=INT_TYPE)
                left = tf.random.uniform(shape=[], minval=0, maxval=img_width-hole_width+1, dtype=INT_TYPE)
                bottom = top + hole_height
                right = left + hole_width

                #assignment is not allowed. so i make grid directly
                height_fill = tf.reshape(tf.cast(tf.range(img_height)<bottom, INT_TYPE)*tf.cast(top<tf.range(img_height), INT_TYPE), [img_height,1,1])
                width_fill = tf.reshape(tf.cast(tf.range(img_width)<right, INT_TYPE)*tf.cast(left<tf.range(img_width), INT_TYPE), [1,img_width,1])
                x = x * tf.cast(1-height_fill*width_fill, FLOAT_TYPE)
                return i + 1, x
            _, self.holes = tf.while_loop(cond, body, (0, self.holes), shape_invariants=None, parallel_iterations=10, back_prop=True, swap_memory=False, maximum_iterations=None, name=None)
            
        img = img * (self.holes) + (1-self.holes) * self.fill_value
        return img
    

class Rotation():
    def __init__(self, p=1.0, max_angle=45, max_shear=0, max_h_shift=0, max_w_shift=0):
        self.p = p
        self.max_angle = max_angle
        self.max_shear = max_shear
        #not implemented
        #self.max_h_shift, self.max_w_shift = max_h_shift, max_w_shift
        
    def __call__(self, inputs, key):
        if key in ["rgb", "depth"]:
            inputs = self.apply(inputs)
        elif key in ["quadrangles"]:
            inputs = self.apply_to_point(inputs)
        return inputs

    def reset_params(self):
        self.rot = self.max_angle * tf.random.normal(shape=[], dtype=tf.float32)
        self.shear = self.max_shear * tf.random.normal(shape=[], dtype=tf.float32)
        self.mat = self.get_mat(self.rot, self.shear)
        #not implemented
        #self.shr = self.max_shear * tf.random.normal(shape=[], dtype=tf.float32)


    def get_params(self):
        return {"rot": self.rot, "shear": self.shear}
    
    def apply(self, img):
        height, width, ch = tf.unstack(tf.shape(img))
        #height, width = img_shape[0], img_shape[1]
        self.height_width = tf.cast(tf.stack([[height,width]]), tf.float32)

        # LIST DESTINATION PIXEL INDICES
        #x = tf.repeat(tf.range(width//2,-width//2,-1), height)
        #y = tf.tile( tf.range(-height//2,height//2),[width])
        x = tf.repeat(tf.range(height//2,-height//2,-1), width)
        y = tf.tile( tf.range(-width//2,width//2),[height])
        ones = tf.ones([height*width],dtype='int32')
        idx = tf.stack( [x,y,ones] )
        
        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = tf.matmul(self.mat, tf.cast(idx,dtype=tf.float32))
        idx2 = tf.cast(idx2,dtype=tf.int32)
        #idx2 = tf.clip_by_value(idx2,-DIM//2+1,DIM//2)
        #idx2 = tf.clip_by_value(idx2[:2], 
        #                        tf.constant([[-width//2+1],[-height//2+1]],tf.int32),
        #                        tf.constant([[width//2],[height//2]],tf.int32))
        idx2 = tf.clip_by_value(idx2[:2], 
                                tf.stack([[-height//2+1],[-width//2+1]]),
                                tf.stack([[height//2],[width//2]]))
        
        # FIND ORIGIN PIXEL VALUES           
        #idx3 = tf.stack( [width//2-idx2[0,], height//2-1+idx2[1,]] )
        idx3 = tf.stack( [height//2-idx2[0,], width//2-1+idx2[1,]] )
        d = tf.gather_nd(img, tf.transpose(idx3))
            
        return tf.reshape(d,[height,width,ch])          
        

    def apply_to_point(self, inputs):# numpoints, 2(x,y)
        #inputs = inputs[:,::-1] - 0.5# x<->y
        #inputs = inputs * self.height_width
        inputs = inputs[:,::-1]# x<->y
        inputs = inputs - self.height_width/2
        
        inputs = tf.concat([inputs, tf.ones_like(inputs[:,0:1])], axis=-1)
        inputs = tf.matmul(inputs, self.mat, transpose_b=True)
        #inputs = tf.transpose(tf.matmul(self.mat, inputs, transpose_b=True))
        inputs = inputs[:,:2] + self.height_width/2#/self.height_width + 0.5
        return inputs[:,::-1]
     
    def get_mat(self, rotation, shear):#, height_zoom, width_zoom, height_shift, width_shift):
        # returns 3x3 transformmatrix which transforms indicies
            
        # CONVERT DEGREES TO RADIANS
        rotation = math.pi * rotation / 180.
        shear = math.pi * shear / 180.
        
        # ROTATION MATRIX
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        one = tf.constant(1,dtype='float32')
        zero = tf.constant(0,dtype='float32')
        #rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        rotation_matrix = tf.stack([[c1,s1,zero], [-s1,c1,zero], [zero,zero,one]])
        # SHEAR MATRIX
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)
        #shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
        shear_matrix = tf.stack([[one,s2,zero], [zero,c2,zero], [zero,zero,one]])
        """    
        # ZOOM MATRIX
        #zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
        zoom_matrix = [[one/height_zoom,zero,zero], [zero,one/width_zoom,zero], [zero,zero,one]]
        
        # SHIFT MATRIX
        #shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
        shift_matrix = [[one,zero,height_shift], [zero,one,width_shift], [zero,zero,one]]
        
        #return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))   
        return tf.matmul(tf.matmul(rotation_matrix, shear_matrix), tf.matmul(zoom_matrix, shift_matrix))   
        """
        return tf.matmul(rotation_matrix, shear_matrix)

