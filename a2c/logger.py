import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class Logger():
    
    def __init__(self):
        self.data = dict()
        self.x_vals = []

    def append(self, new_data, x_val=None):
        """
        Appends data to the data dict.

        new_data - dictionary with data pts
        x_val - value of x axis to associate with the data contained in
                new_data
        """
        for key in new_data.keys():
            if key in self.data:
                self.data[key].append(new_data[key])
            else:
                self.data[key] = [new_data[key]]
        if x_val is not None:
            self.x_vals.append(x_val)

    def log_data(self):
        pass
        
    def make_plots(self, save_name='', xlabel='Frames'):

        l = len(self.data[list(self.data)[0]])
        if len(self.x_vals) != l:
            self.x_vals = np.arange(l)

        if save_name != "":
            save_name = save_name+"_"

        for key in self.data.keys():
            plt.plot(self.x_vals, self.data[key]) 
            plt.xlabel(xlabel)
            plt.ylabel(key)
            plt.title(key)
            plt.savefig(save_name+key+".png")
            plt.clf()
            
