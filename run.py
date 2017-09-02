from deepretina.models import convnet, ln, sequential
from deepretina.core import train
from keras.optimizers import RMSprop
from deepretina.experiments import Experiment
from deepretina.io import KerasMonitor, main_wrapper

def fit_ln(cells, train_stimuli, exptdate, stim_shape, l2=1e-3, readme=None):
    """Fits an LN model using keras"""
    ncells = len(cells)
    batchsize = 5000

    # get the layers
    layers = ln(stim_shape, ncells, weight_init='normal', l2_reg=l2)

    # compile it
    model = sequential(layers, RMSprop(lr=1e-4), loss='poisson')

    # load experiment data
    test_stimuli = ['whitenoise', 'naturalscene']
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize, nskip=6000)

    # create a monitor that keeps track of progress
    monitor = KerasMonitor('ln', model, data, readme, save_every=20)

    # train
    train(model, data, monitor, num_epochs=30)

    return model
def fit_convnet(cells, train_stimuli, exptdate, nclip=0, readme=None):
    """Main script for fitting a convnet

    author: Niru Maheswaranathan
    """

    stim_shape = (30, 50, 50)
    ncells = len(cells)
    batchsize = 500#5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(15, 7), weight_init='normal',
                     l2_reg_weights=(0.01, 0.01, 0.01),
                     l1_reg_activity=(0.0, 0.0, 0.001),
                     dropout=(0.1, 0.0))

    # compile the keras model

    model = sequential(layers, 'adam', loss='poisson')
    print(model.summary())
    # load experiment data
    test_stimuli = ['whitenoise',]#, 'naturalscene']
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize, nskip=nclip)

    # create a monitor to track progress
    monitor = KerasMonitor('convnet', model, data, readme, save_every=20)
    print("monitor has been shutdown by fxy")
    # train
    train(model, data, None, num_epochs=50)

    return model
if __name__=='__main__':
    fit_convnet([2,],['whitenoise_temp'],'train')