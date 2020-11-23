import matplotlib.pyplot as plt
from scipy.io import savemat
import os
import numpy as np

class Visualization(object):
    def __init__(self, figs_pool, save_path,  **kwargs):
        self.figures = dict(figs_pool)
        self.save_path = save_path


    def reset_states(self):
        for fig in self.figures.values():
            fig.reset_states()

    def update_state(self, data):
        if data is None: return

        for name, d in data.items():
            if name in self.figures.keys():
                self.figures[name].update_state(d)

    def plot(self, epoch, save=None):
        for name, fig in self.figures.items():
            if isinstance(fig, Plot):
                save_name = "{}.png".format(name)
            else:
                save_name = "{}_{:05d}.png".format(name, epoch)

            fig.plot(save=save, save_path=self.save_path, save_name=save_name)

class Figure(object):

    def __init__(self, name='fig', **kwargs):
        self.name = name
        self.fig_data = list()

    def reset_states(self):
        self.fig_data = list()

    def aggregate_data(self):
        return np.concatenate(self.fig_data, axis=0)

    def update_state(self, data):
        self.fig_data.append(data)

    def plot(self, save=None):
        pass

class Plot(Figure):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def aggregate_data(self):
        data = np.array(self.fig_data)
        return np.squeeze(data)

    def reset_states(self):
        pass

    def plot(self, save=None, save_path="./", save_name="fig.png"):

        data = self.aggregate_data()

        plt.figure()
        plt.plot(data)
        plt.title(self.name)

        if save:
            plt.savefig(os.path.join(save_path, save_name))
            savemat(os.path.join(save_path, self.name + '_raw_data.mat'),
                    {"data": data})
        plt.close()
