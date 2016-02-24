"""
Main script for training deep retinal models

"""

from __future__ import absolute_import
from deepretina.models import sequential, convnet, ln, train
from deepretina.experiments import Experiment, _loadexpt_h5
from deepretina.io import Monitor, main_wrapper
import numpy as np
from keras.optimizers import Adam, RMSprop


@main_wrapper
def fit_ln(cells, train_stimuli, exptdate, readme=None):
    """Fits an LN model using keras"""
    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the layers
    layers = ln(stim_shape, ncells, weight_init='normal', l2_reg=1.0)

    # compile it
    model = sequential(layers, RMSprop(lr=1e-4), loss='sub_poisson_loss')

    # load the STAs
    stas = []
    h5file = _loadexpt_h5(exptdate, train_stimuli[0])
    for ci in cells:
        key = 'cell{:02}'.format(ci + 1)
        stas.append(np.array(h5file['stas'][key]).ravel())

    # specify the initial weights using the STAs
    W = np.vstack(stas).T
    b = np.zeros(W.shape[1])
    model.layers[1].set_weights([W, b])

    # load experiment data
    test_stimuli = ['whitenoise', 'naturalscene']
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

    # create a monitor
    monitor = Monitor('ln', model, data, readme, save_every=10)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


@main_wrapper
def fit_convnet(cells, train_stimuli, exptdate, readme=None):
    """Main script for fitting a convnet

    author: Niru Maheswaranathan
    """

    stim_shape = (40, 50, 50)
    ncells = len(cells)
    batchsize = 5000

    # get the convnet layers
    layers = convnet(stim_shape, ncells, num_filters=(8, 16),
                     filter_size=(13, 13), weight_init='normal', l2_reg=0.1)

    # compile the keras model
    model = sequential(layers, 'adam')

    # load experiment data
    test_stimuli = ['whitenoise', 'naturalscene']
    data = Experiment(exptdate, cells, train_stimuli, test_stimuli, stim_shape[0], batchsize)

    # create a monitor to track progress
    monitor = Monitor('convnet', model, data, readme, save_every=5)

    # train
    train(model, data, monitor, num_epochs=100)

    return model


if __name__ == '__main__':

    # mdl = fit_ln([0, 1, 2, 3, 4, 5], ['naturalscene'], '15-10-07', description='LN models w/ sta initialization (ns)')
    # mdl = fit_ln([0, 1, 2, 3, 4, 5], ['whitenoise'], '15-10-07', description='LN models w/ sta initialization (wn)')

    # mdl = fit_ln(list(range(37)), ['whitenoise'], 'all-cells', description='LN models on whitenoise')
    # mdl = fit_ln(list(range(37)), ['naturalscene'], 'all-cells', description='LN models on naturalscene')

    gc_151121a = [6, 10, 12, 13]
    mdl = fit_ln(gc_151121a, ['naturalscene'], '15-11-21a', description='LN models w/ sta initialization (ns)')
    mdl = fit_ln(gc_151121a, ['whitenoise'], '15-11-21a', description='LN models w/ sta initialization (wn)')

    gc_151121b = [0, 1, 3, 4, 5, 8, 9, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    mdl = fit_ln(gc_151121b, ['naturalscene'], '15-11-21b', description='LN models w/ sta initialization (ns)')
    mdl = fit_ln(gc_151121b, ['whitenoise'], '15-11-21b', description='LN models w/ sta initialization (wn)')