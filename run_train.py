import os
import sys
import argparse
import numpy as np
import skimage.io
import sklearn.metrics

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam

import util
import cnn

if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Train CNN with MI learning.' )
    parser.add_argument('--in_dir', '-i', required=True, help='input directory' )
    parser.add_argument('--out_dir', '-o', required=True, help='output directory' )
    parser.add_argument('--in_model', help='Filename of model to initialize with' )
    parser.add_argument('--out_model', help='Filename of model to save' )
    parser.add_argument('--save_results', help='Save predictions on test set to file' )
    parser.add_argument('--test_only', action='store_true', help='Test only' )
    #parser.add_argument('--prior', action='store_true', help='Use prior probabilities from training set' )
    parser.add_argument('--fold', '-f', required=True, help='Cross validation fold #' )
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all' )
    parser.add_argument('--model', '-m', required=True, help='CNN model' )
    parser.add_argument('--crop', '-c', help='Crop size' )
    parser.add_argument('--test_crop', help='Test crop size (default: 3000)' )
    parser.add_argument('--mi', help='MI aggregation type (mean, quantile)' )
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16' )
    parser.add_argument('--rate', '-r', help='Learning rate; if cyclic "r0,r1"' )
    parser.add_argument('--lr_range', help='Learning rate test; "start,stop,steps"' )
    parser.add_argument('--batch_size', '-b', help='Batch size' )
    parser.add_argument('--epochs', '-e', help='Epochs' )
    parser.add_argument('--init_epoch', help='Initial epoch' )
    parser.add_argument('--mask', action='store_true', help='use mask' )
    parser.add_argument('--freeze', action='store_true', help='freezer lower layers' )
    parser.add_argument('--balance', action='store_true', help='balance training samples by class labels' )
    parser.add_argument('--n_jobs', help='number of parallel threads' )
    parser.add_argument('--gpu', '-g', help='selected GPU' )
    args = parser.parse_args()
    src_dir = args.in_dir
    if len(src_dir) > 1 and src_dir[-1] != '/':
        src_dir += '/'
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    in_model = args.in_model
    out_model = args.out_model
    save_results = args.save_results
    test_only = args.test_only
    fold = args.fold
    categories = args.cat
    model_name = args.model
    crop = args.crop
    if crop is not None:
        crop = crop.split(',')
        if len(crop) == 1:
            crop = (int(crop[0]),int(crop[0]))
        else:
            crop = (int(crop[0]),int(crop[1]))
    test_crop = args.test_crop
    if test_crop is not None:
        test_crop = test_crop.split(',')
        if len(test_crop) == 1:
            test_crop = (int(test_crop[0]),int(test_crop[0]))
        else:
            test_crop = (int(test_crop[0]),int(test_crop[1]))
    mi_type = args.mi
    quantiles = args.quantiles
    lr = args.rate
    if lr is not None:
        lr = lr.split(',')
        lr = [float(r) for r in lr]
    lr_range = args.lr_range
    if lr_range is not None:
        lr_range = args.lr_range.split(',')
        if len(lr_range) == 1:
            lr_range = float(lr_range[0])
        else:
            lr_range = [float(r) for r in lr_range]
    batch_size = args.batch_size
    if batch_size is not None:
        batch_size = int(batch_size)
    epochs = args.epochs
    if epochs is not None:
        epochs = int(epochs)
    init_epoch = int(args.init_epoch) if args.init_epoch is not None else 0
    use_mask = args.mask
    freeze = args.freeze
    balance = args.balance
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else 1
    gpu = args.gpu

    # load filenames and labels
    sample_images = util.load_sample_images( out_dir )
    samples,cats,labels = util.load_labels( out_dir )

    max_inst = max([len(si) for si in sample_images.values()])
    print('max instances',max_inst)

    # load filenames and labels
    image_list = util.load_image_list( out_dir )
    if use_mask:
        mask_list = util.load_mask_list( out_dir )
        sample_masks = util.load_sample_masks( out_dir )
    else:
        mask_list = [None]*len(image_list)
        sample_masks = {}
        
    if categories is None:
        categories = cats
    else:
        categories = categories.split(',')
        
    # get labels for list of categories
    label_names = []
    new_labels = np.zeros((labels.shape[0],len(categories)),dtype='int')
    for i,cat in enumerate(categories):
        c = np.where(cats==cat)[0][0]
        ln = np.unique([l[c] for l in labels])
        ln.sort()
        ln = list(ln)
        if '' in ln:
            del ln[ln.index('')]
        label_names.append( ln )
        new_labels[:,i] = np.array([ ln.index(l) if l in ln else -1 for l in labels[:,c] ])
    labels = new_labels
    cats = categories

    # create list of class names for each category
    classes = []
    for c in range(len(cats)):
        cl = np.unique(labels[:,c])
        np.sort(cl)
        if cl[0] == -1:
            cl = cl[1:]
        classes.append((cats[c],cl))
    print(classes)

    # split into train/test sets
    if fold is not None:
        idx_train_val_test = util.load_cv_files( out_dir, samples, 'fold'+str(fold)+'.csv' )[0]
        print(idx_train_val_test)
        idx_train  = idx_train_val_test[0]
        idx_val = idx_train_val_test[1]
        idx_test = idx_train_val_test[2]
    else:
        idx_train = np.arange(len(samples))
        idx_val = np.arange(0)
        idx_test = np.arange(0)

    # drop samples with missing label for all categories
    idx_train = np.array( [ i for i in idx_train if (labels[i,:]!=-1).sum()>0 ] )
    idx_val = np.array( [ i for i in idx_val if (labels[i,:]!=-1).sum()>0 ] )
    #idx_test = np.array( [ i for i in idx_test if (labels[i,:]!=-1).sum()>0 ] )

    print('GPU available: ',tf.test.is_gpu_available())
    if gpu is not None:
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        #session = tf.Session(config=config)
        
        tf.keras.backend.set_floatx('float16')
        print('Setting GPU to %s'%gpu)
        #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        #tf.device('/device:GPU:'+gpu)
        #gpus = tf.config.experimental.list_physical_devices('GPU')
        if False:#gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
                sys.exit(1)

    base_model,preprocess_input = cnn.load_base_model( model_name )
    model = cnn.add_mi_layer( base_model, classes, mi_type, use_mask=use_mask )

    print(model.summary())

    if in_model is not None:
        print('Initializing with '+in_model)
        model.load_weights(out_dir+in_model)

    if not test_only:
        # freeze lower?
        if freeze:
            for layer in model.layers:
                if 'softmax' not in layer.name:
                    layer.trainable = False

        cat_loss = cnn.categorical_crossentropy_missing
        cat_acc = cnn.categorical_accuracy_missing

        # train and val generators
        gen_train = cnn.ImageSequence( src_dir, [sample_images[samples[s]] for s in idx_train], labels[idx_train,:], classes, crop, batch_size, preprocess_input, mask_list=[sample_masks[samples[s]] for s in idx_train] if use_mask else None, random=True, balance=balance )
        gen_val = cnn.ImageSequence( src_dir, [sample_images[samples[s]] for s in idx_val], labels[idx_val,:], classes, crop, batch_size, preprocess_input, mask_list=[sample_masks[samples[s]] for s in idx_val] if use_mask else None, random=False )#, sample_instances=max_inst )

        model.compile( optimizer=Nadam(lr=lr[0]), loss=[cat_loss]*len(model.outputs), metrics=[cat_acc] )

        callbacks = []
        if lr_range is not None:
            # lr range test
            from keras_one_cycle_clr.keras_one_cycle_clr.lr_range_test import LrRangeTest
            lrrt_cb = LrRangeTest( lr_range=(lr_range[0],lr_range[1]), wd_list=[0], steps=lr_range[2], batches_per_step=100//batch_size, validation_data=gen_val, batches_per_val=50, verbose=True, custom_objects={'categorical_crossentropy_missing':cat_loss,'categorical_accuracy_missing':cat_acc} )
            n_epochs = lrrt_cb.find_n_epoch(gen_train)
            model.fit_generator( gen_train, epochs=n_epochs, max_queue_size=5, workers=n_jobs, callbacks=[lrrt_cb] )
            lrrt_cb.plot()
            sys.exit(0)

        if len(lr) > 1 :
            # cyclic lr
            from keras_one_cycle_clr.keras_one_cycle_clr.cyclic_lr import CLR
            from keras_one_cycle_clr.keras_one_cycle_clr.utils import plot_from_history
            clr_cb = CLR( cyc=lr[2], lr_range=(lr[0],lr[1]), momentum_range=(0.95, 0.85), verbose=True, amplitude_fn=lambda x: np.power(1.0/3, x) )
            clr_hist = model.fit_generator( gen_train, epochs=epochs, validation_data=gen_val, max_queue_size=5, workers=n_jobs, shuffle=True )
            plot_from_history(clr_hist)
        else:
            # constant lr
            model.fit_generator( gen_train, epochs=epochs, verbose=2, validation_data=gen_val, max_queue_size=5, workers=n_jobs, shuffle=True, initial_epoch=init_epoch )

        # save model
        if out_model is not None:
            model.save( out_dir+out_model )
        
    # predict on test data
    if save_results is not None or test_only:
        # put all instances into list
        test_images = [ sample_images[samples[s]] for s in idx_test ]
        if use_mask:
            test_masks = [ sample_masks[samples[s]] for s in idx_test ]
        test_labels = labels[idx_test,:]
        test_labels = np.array([ l for images,l in zip(test_images,test_labels) for s in images ])
        test_inst2sample = np.array([ s for s in range(len(test_images)) for j in test_images[s] ])
        test_images = [ [inst] for im in test_images for inst in im ]
        if use_mask:
            test_masks = [ [inst] for im in test_masks for inst in im ]
        test_labels = np.array(test_labels)

        gen_test = cnn.ImageSequence( src_dir, test_images, test_labels, classes, test_crop, 1, preprocess_input, mask_list=test_masks if use_mask else None, random=False )#, sample_instances=max_inst )
        p = model.predict_generator( gen_test, steps=len(test_images), max_queue_size=5, workers=n_jobs, use_multiprocessing=True )
        if type(p) is not list:
            p = [p]
        
        # aggregate across instances
        p = np.array( [ [ pcat[test_inst2sample==s,:].mean(axis=0) for s in range(len(idx_test)) ] for pcat in p ] )

        for c,cat in enumerate(cats):
            print(cat)
            pcat = p[c]
            lcat = labels[idx_test,c]

            # drop samples with missing label
            idx = (lcat!=-1)
            pcat = pcat[idx]
            dcat = pcat.argmax(axis=1)
            lcat = lcat[idx]

            binary = len(classes[c][1]) <= 2

            if binary:
                print('auc',sklearn.metrics.roc_auc_score( lcat, pcat[:,1] ))
                print('ap',sklearn.metrics.average_precision_score( lcat, pcat[:,1] ))

            #if prior is not None:
            print('No prior')
            print('accuracy',sklearn.metrics.accuracy_score( lcat, dcat ))
            if binary:
                print('precision',float( ( np.logical_and(dcat == 1, dcat == lcat) ).sum() ) / ( dcat == 1 ).sum())
                print('sensitivity/recall/tpr',float( ( np.logical_and(lcat == 1, dcat == 1) ).sum() ) / ( lcat == 1 ).sum())
                print('specificity/tnr',float( ( np.logical_and(lcat == 0, dcat == 0) ).sum() ) / ( lcat == 0 ).sum())
                print('fpr',float( ( np.logical_and(lcat == 0, dcat == 1) ).sum() ) / ( lcat == 0 ).sum())
                print('fnr',float( ( np.logical_and(lcat == 1, dcat == 0) ).sum() ) / ( lcat == 1 ).sum())
            print('f1',sklearn.metrics.f1_score( lcat, dcat, average='macro' ))
            #print(sklearn.metrics.classification_report( lcat, pcat, target_names=label_names[c] ))
            print('confusion',[int(l) for l in label_names[c]])
            print(sklearn.metrics.confusion_matrix( lcat, dcat ))

            if True:#prior is not None:
                prob = np.array([ (labels[idx_train,c]==l).sum() for l in classes[c][1] ] )
                prob = prob/prob.sum()
                
                print('Prior',prob)
                dcat = np.argmax(pcat*prob,axis=1)
                print(pcat)
                print(dcat)
                print('accuracy',sklearn.metrics.accuracy_score( lcat, dcat ))
                if binary:
                    print('precision',float( ( np.logical_and(dcat == 1, dcat == lcat) ).sum() ) / ( dcat == 1 ).sum())
                    print('sensitivity/recall/tpr',float( ( np.logical_and(lcat == 1, dcat == 1) ).sum() ) / ( lcat == 1 ).sum())
                    print('specificity/tnr',float( ( np.logical_and(lcat == 0, dcat == 0) ).sum() ) / ( lcat == 0 ).sum())
                    print('fpr',float( ( np.logical_and(lcat == 0, dcat == 1) ).sum() ) / ( lcat == 0 ).sum())
                    print('fnr',float( ( np.logical_and(lcat == 1, dcat == 0) ).sum() ) / ( lcat == 1 ).sum())
                    print('f1',sklearn.metrics.f1_score( lcat, dcat, average='macro' ))
                print('confusion',[int(l) for l in label_names[c]])
                print(sklearn.metrics.confusion_matrix( lcat, dcat ))
            
        # save
        if save_results:
            header = [['ID']]+[ [ cats[c]+'_gt' ] + [ cats[c]+'_'+ln for ln in label_names[c] ] for c in range(len(cats)) ]
            header = [ inner for outer in header for inner in outer ]
            res = np.vstack( [ header, np.hstack([np.expand_dims(samples[idx_test],axis=1)]+[np.hstack((np.expand_dims(labels[idx_test,c].astype(str),axis=1),p[c].astype(str))) for c in range(labels.shape[1])]) ] )
            print('Saving '+out_dir+save_results)
            np.savetxt(out_dir+save_results,res,fmt='%s',delimiter=',')
    
