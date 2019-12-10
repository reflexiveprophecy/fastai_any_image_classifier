import os
import re
import numpy as np
import copy 
from fastai.vision import verify_images, ImageDataBunch, imagenet_stats, get_transforms, models, \
ClassificationInterpretation, cnn_learner, load_learner, open_image, torch, defaults
from fastai.metrics import error_rate


data_directory = './image_data/'
files_in_directory = os.listdir(data_directory)


def image_verification(delete = True, max_size = 500):
    '''this function helps to verify all the images in the directory'''

    #filtering out any system files or models that are not image data folders
    classes_in_directory = filter(lambda x: ('.' not in x) & ('models' not in x), files_in_directory)
    
    #loop through each image data folder to delete any images that are not openable
    for image_class in classes_in_directory:
        print(image_class)
        verify_images(data_directory + image_class, delete=delete, max_size=max_size)

    return None


def image_learner(valid_pct = 0.2, size = 224, model = models.resnet34, num_workers = 8, bs = 32, 
                    max_rotate = 20, max_zoom = 1.3, max_lighting = 0.4, max_warp = 0.4, p_affine = 1,
                    p_lighting = 1.):
    '''this function helps to prepare the data for training purposes'''
    
    np.random.seed(2)
    #setting the image augmentation
    tfms = get_transforms(max_rotate=max_rotate, max_zoom=max_zoom, max_lighting=max_lighting, max_warp=max_warp,
                      p_affine=p_affine, p_lighting=p_lighting)
    #instantiate an ImageDataBunch class
    data = ImageDataBunch.from_folder(data_directory, train=".", valid_pct=valid_pct, bs = bs,
        ds_tfms=tfms, size=size, num_workers=num_workers).normalize(imagenet_stats)
    print('the following images are the examples of prepared training data')
    data.show_batch(rows=3, figsize=(7,8))
    print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))
    #instantiate a learner here to pass down to other functions
    learn = cnn_learner(data, model, metrics = error_rate)

    return learn


def frozen_training(learner, num_of_cycles = 4):
    '''training the model with last two layers frozen and unfreeze the entire model'''
    learner.fit_one_cycle(num_of_cycles)
    learner.save('./frozen-model')
    return learner


def tune_learning_rate(learner, start_lr = 1e-5, end_lr = 1e-1):
    learner.unfreeze()
    learner.lr_find(start_lr = start_lr, end_lr = end_lr)
    #generate a learning rate plot in the folder to manually determine the range of desired learning rates 
    fig = learner.recorder.plot(return_fig = True)
    #saving the learning rate chart as learning_rate_graph.png
    fig.savefig(data_directory + 'models/' +'learning_rate_graph.png')
    return None


def unfrozen_training(learner):
    '''transfer learning using resnet34'''
    learner.unfreeze()
    #please input the desired learning rates based on the learning rate graph generated
    refined_learning_rate = input('please input the selected learning rate min and max, separated by comma, i.e. 3e-5,3e-4: ')
    learning_rate_min, learning_rate_max = [float(x) for x in refined_learning_rate.split(',')]
    learner.fit_one_cycle(6, max_lr=slice(learning_rate_min, learning_rate_max))
    learner.save('./unfrozen-model')
    learner.export()

    return learner


def model_evaluation(learner):
    '''evaluate the trained model'''
    learn = learner.load('unfrozen-model')
    interp = ClassificationInterpretation.from_learner(learn)
    print(interp.plot_confusion_matrix())
    return None


def main():
    '''wrapping all the functions and variables together'''
    image_verification()
    frozen_model = frozen_training(image_learner())
    print(frozen_model.summary())
    tune_learning_rate(frozen_model)
    unfrozen_model = unfrozen_training(frozen_model)
    print(unfrozen_model.summary())
    model_evaluation(unfrozen_model)


if __name__ == '__main__':
    main()




