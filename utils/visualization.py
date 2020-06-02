#Visualize Acc & Loss
import pygal

import sys
sys.path.append('./')

######################################################

def visualize(h, name):
    graphacc = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphacc.title = 'Accuracy'
    graphacc.add('Training Acc', h.history['accuracy'])
    graphacc.add('Testing Acc', h.history['val_accuracy'])

    graphacc.render_to_file('assets/visual/' + name + '-Accuracy.svg')

    ######################################################

    graphloss = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphloss.title = 'Loss'
    graphloss.add('Training Loss', h.history['loss'])
    graphloss.add('Testing Loss', h.history['val_loss'])

    graphloss.render_to_file('assets/visual/' + name + '-Loss.svg')

    return