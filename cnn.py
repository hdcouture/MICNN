import numpy as np
import skimage.transform

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Flatten, Lambda, Reshape, Multiply, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.inception_v3 import InceptionV3
#from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
#from tensorflow.keras.applications.xception import Xception
#from tensorflow.keras.applications.densenet import DenseNet201
    
_EPSILON = 10e-8

class Softmax4D(Layer):
    '''Apply softmax fully convolutionally'''

    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape(self, input_shape):
        axis_index = self.axis % len(input_shape)
        return input_shape
                      #if i != axis_index ])

def load_base_model( model_name ):
    '''Load pre-trained model.

    Parameters:
    model_name - one of ResNet50, VGG16, InceptionV3, InceptionResNetV2, Xception, DenseNet201

    Returns:
    TF model with pre-trained weights and no softmax layer
    '''

    max_dim = None
    input_tensor = Input(shape=(max_dim,max_dim,3))
    if model_name.lower() == 'resnet50':
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(input_shape=(max_dim,max_dim,3),include_top=False,weights='imagenet')
    elif model_name.lower() == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        base_model = VGG16(input_shape=(max_dim,max_dim,3),include_top=False,weights='imagenet')
    elif model_name.lower() == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        base_model = InceptionV3(input_tensor=input_tensor,include_top=False,weights='imagenet')
    elif model_name.lower() == 'inceptionresnetv2':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        base_model = InceptionResNetV2(input_tensor=input_tensor,include_top=False,weights='imagenet')
    elif model_name.lower() == 'xception':
        from tensorflow.keras.applications.xception import Xception
        from tensorflow.keras.applications.xception import preprocess_input
        base_model = Xception(input_tensor=input_tensor,include_top=False,weights='imagenet')
    elif model_name.lower() == 'densenet201':
        from tensorflow.keras.applications.densenet import DenseNet201
        from tensorflow.keras.applications.densenet import preprocess_input
        base_model = DenseNet201(input_tensor=input_tensor,include_top=False,weights='imagenet')
    else:
        print('Error: unsupported model')
        sys.exit(1)

    return base_model,preprocess_input

def add_mi_layer( orig_model, classes, mi_type, quantiles=16, use_mask=False ):
    '''Add MI layer to existing model.

    Parameters:
    orig_model - TF model (typically pre-trained)
    classes - list of classes and labels, e.g., [('class1',[0,1]),('class2',[0,1,2])]
    mi_type - type of MI aggregation: None (default, mean pool features), mean, quantile
    quantiles - number of quantiles to use if mi_type is 'quantile' (default 16)
    use_mask - whether to apply mask to image when pooling
    '''

    top_layer = orig_model.output
    
    if use_mask:
        # downsize mask to match image downsize operations
        
        shape = orig_model.input_shape
        mask_input = Input(shape=(shape[1],shape[2],1))
        xmask = mask_input

        done_layers = []
        for layer in orig_model.layers:
            config = layer.get_config()
            if 'padding' in config:
                padding = config['padding']
            else:
                padding = 'valid'
            if 'strides' in config:
                strides = config['strides']
            else:
                strides = None
            if 'pool_size' in config:
                pool_size = config['pool_size']
            elif 'kernel_size' in config:
                pool_size = config['kernel_size']
            else:
                pool_size = 1
            if type(layer.input) is list:
                for l in layer.input:
                    if l.name in done_layers:
                        continue
                else:
                    done_layers.append(l.name)
            else:
                if layer.input.name in done_layers:
                    continue
                else:
                    done_layers.append(layer.input.name)
            if pool_size == 1 and ( strides is None or strides == (1,1) ) and padding in ['valid','same']:
                continue
            if strides == (1,1) and padding in ['valid','same']:
                continue
            if type(padding) is not str:
                xmask = ZeroPadding2D(padding)(xmask)
                padding = 'valid'
            xmask = AveragePooling2D(pool_size,strides,padding)(xmask)

    if use_mask:
        # normalized to sum to one
        xmask_norm = Lambda(lambda z: z / (K.sum(z, axis=(1,2), keepdims=True)+_EPSILON), output_shape=lambda input_shape:input_shape, name='mask_norm')(xmask)
        
    # loop through classes
    outputs = []
    for c,cl in classes:
        if mi_type is None:
            x = top_layer
        else:#if mi_type == 'mean':
            x = Conv2D(len(cl),(1,1),name='softmaxfc_'+c)(top_layer)
            x = Softmax4D(axis=1,name='softmax_'+c)(x)
            
        if not use_mask:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Multiply(name='multiply_'+c)([x,xmask_norm])
            x = Lambda( lambda z: K.sum(z, axis=(1,2), keepdims=False), output_shape=lambda input_shape:(input_shape[0],input_shape[3]), name='sum_'+c)(x)

        if mi_type is None:
            x = Dense(len(cl), activation='softmax', name='softmax_'+c)(x)

        outputs.append(x)

    if use_mask:
        model = Model(inputs=[orig_model.input,mask_input], outputs=outputs)
    else:
        model = Model(inputs=orig_model.input, outputs=outputs)

    return model

def categorical_crossentropy_missing(target, output):
    """Loss function that ignores samples with a missing label (all 0s)."""

    target = K.cast(target,'float32')
    output = K.cast(output,'float32')
    # scale preds so that the class probas of each sample sum to 1
    output /= (K.sum(output,axis=1, keepdims=True)+_EPSILON)
    # avoid numerical instability with _EPSILON clipping
    output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    select = K.cast( K.greater_equal(K.max(target,axis=1),0.5), 'float32' )
    ce = -K.sum(target * K.log(output), axis=1)
    return K.sum( ce * select ) / (K.sum(select)+_EPSILON)

def categorical_accuracy_missing(y_true, y_pred):
    """Metric to calculate accuracy while ignoring samples with a missing label."""
    
    select = K.cast( K.greater_equal(K.max(y_true,axis=1),0.5), 'float32' )
    return K.sum(K.cast(K.equal(K.argmax(y_true, axis=1),
                                 K.argmax(y_pred, axis=1)),'float32')*select) / (K.sum(select)+_EPSILON)

class ImageSequence(Sequence):
    '''Generate image,label pairs for training.

    Parameters:
    image_dir - directory where images are stored
    image_list - list of lists of image files; each list is for a different sample
    labels - numpy array of labels for each sample
    classes - list of classes and labels, e.g., [('class1',[0,1]),('class2',[0,1,2])]
    crop - size of image to randomly crop
    batch_size - batch size for training
    preprocess_input - function for preprocessing images
    sample_instances - max number of instances to use from each sample
    mask_list - same as image_list but for mask files (if needed)
    random - whether to sample randomly
    balance - draw samples so that class labels are balanced
    '''

    def __init__(self, image_dir, image_list, labels, classes, crop, batch_size, preprocess_input, sample_instances=1, mask_list=None, random=True, balance=False):
        self.image_dir = image_dir
        self.image_list = image_list
        self.labels = labels
        self.classes = classes
        self.crop = crop
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.sample_instances = sample_instances
        self.mask_list = mask_list
        self.random = random
        self.balance = balance

    def __len__(self):
        return int(np.ceil(np.sum([min(len(im),self.sample_instances) for im in self.image_list]) / float(self.batch_size)))

    def __getitem__(self, idx):

        x_img = []
        x_mask = []
        y = []
        for c in range(self.labels.shape[1]):
            y += [[]]
        for i in range(idx*self.batch_size,(idx+1)*self.batch_size):
            if i >= len(self.image_list):
                break

            if self.balance:
                # choose image to balance classes
                c = np.random.randint(0,len(self.classes))
                l = np.random.choice(self.classes[c][1])
                i = np.random.choice(np.where(self.labels[:,c]==l)[0])
            
            # randomly choose image from set
            ninst = len(self.image_list[i])
            rand_inst = np.random.choice(np.arange(ninst),min(self.sample_instances,ninst),replace=False)
            for j in rand_inst:
                img_fn = self.image_list[i][j]
                if self.mask_list is not None:
                    mask_fn = self.mask_list[i][j]

                # read image
                img = image.load_img( self.image_dir+img_fn )
                img = image.img_to_array( img )

                # read mask
                if self.mask_list is not None:
                    mask = image.load_img( self.image_dir+mask_fn )
                    mask = image.img_to_array( mask )
                    mask = mask[:,:,0]
                    mask /= mask.max()

                # TODO: pad images with zero to fit largest size
                if self.crop[0] > img.shape[0] or self.crop[1] > img.shape[1]:
                    pad = ( (self.crop[0]-img.shape[0])//2, (self.crop[1]-img.shape[1])//2 )
                    img = skimage.util.pad( img, pad_width=pad, mode='constant' )

                # random transformation
                if self.random:
                    if np.random.randint(0,2) == 1:
                        img = np.fliplr(img)
                        if self.mask_list is not None:
                            mask = np.fliplr(mask)
                    rot = np.random.random()*360.0
                    img = skimage.transform.rotate( img, rot, resize=False, mode='constant', cval=img.max() ).astype('float16')
                    if self.mask_list is not None:
                        mask = skimage.transform.rotate( mask, rot, resize=False, mode='constant', order=0, cval=0 ).astype('float16')

                # random crop
                if self.random:
                    it = 0
                    thresh = 0.5
                    while True:
                        top = np.random.randint(0,img.shape[0]-self.crop[0]+1)
                        left = np.random.randint(0,img.shape[1]-self.crop[1]+1)
                        if self.mask_list is not None:
                            mask_sum = mask[top:top+self.crop[0],left:left+self.crop[1]].mean()
                            if mask_sum > thresh:
                                break
                        else:
                            break
                        it += 1
                        if (it+1) % 100 == 0:
                            thresh /= 2
                        if it > 1e6:
                            print('stuck in loop in find crop with > 50% in mask')
                            sys.exit(1)
                else:
                    top = (img.shape[0]-self.crop[0])//2
                    left = (img.shape[1]-self.crop[1])//2
                img = img[top:top+self.crop[0],left:left+self.crop[1],:]
                if self.mask_list is not None:
                    mask = mask[top:top+self.crop[0],left:left+self.crop[1]]
                    shape1 = mask.shape
                    mask = np.expand_dims( mask, axis=0 )
                    mask = np.expand_dims( mask, axis=3 )

                img = np.expand_dims( img, axis=0 )
                img = self.preprocess_input( img )
                x_img.append(img)

                if self.mask_list is not None:
                    x_mask.append(mask)

                for c in range(len(y)):
                    y[c].append(self.labels[i,c])

        x_img = np.concatenate(x_img,axis=0)

        # convert labels to categorical
        y_cat = []
        for yi,cl in zip(y,self.classes):
            yi = np.array(yi,dtype='float16')
            yi2 = to_categorical(yi,len(cl[1]))
            yi2[yi==-1,:] = 0
            y_cat.append(yi2)

        if self.mask_list is not None:
            x_mask = np.concatenate(x_mask,axis=0)
            return (x_img,x_mask),y_cat
        return x_img,y_cat
