import numpy as np
import visdom


class Visualizer():
    
    def __init__(self,display_port):

        self.vis = visdom.Visidom(port = display_port) 


    def display_current_images():
        pass


    def plot_current_errors():
        pass


    def plot_current_metrics():
        pass



