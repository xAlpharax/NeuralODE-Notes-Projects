#Visualize Acc & Loss
import pygal

import sys
sys.path.append('./')

##################################################
#############################################################
#############################################

def visualize(h, name):
    graphacc = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphacc.title = 'Accuracy'
    graphacc.add('Training Acc', h.history['accuracy'])
    graphacc.add('Testing Acc', h.history['val_accuracy'])

    graphacc.render_to_file('assets/visuals/' + name + '-Accuracy.svg')

    #############################################################

    graphloss = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphloss.title = 'Loss'
    graphloss.add('Training Loss', h.history['loss'])
    graphloss.add('Testing Loss', h.history['val_loss'])

    graphloss.render_to_file('assets/visuals/' + name + '-Loss.svg')

    return

##################################################
#############################################################
#############################################

def customvis(name, TrA, TeA, TrL, TeL):
    graphacc = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphacc.title = 'Accuracy'
    graphacc.add('Training Acc', TrA)
    graphacc.add('Testing Acc', TeA)

    graphacc.render_to_file('assets/celebvisuals/' + name + '-Accuracy.svg')

    #############################################################

    graphloss = pygal.Line() #style=pygal.style.RotateStyle('#9e6ffe'), interpolate='hermite'
    graphloss.title = 'Loss'
    graphloss.add('Training Loss', TrL)
    graphloss.add('Testing Loss', TeL)

    graphloss.render_to_file('assets/celebvisuals/' + name + '-Loss.svg')

    return

##################################################
#############################################################
#############################################