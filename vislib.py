# %%
"""
library for visualization 
"""

import numpy as np
import matplotlib.pyplot as plt


class BlitPlot:
    """
    A class for creating and updating blitting plots.

    Parameters:
    - data: The data to be plotted.
    - plot_type: The type of plot to create ("line", "event", "image", "trajectory").
    - ax: The axes object to plot on. If None, a new axes object will be created.
    - fig: The figure object to plot on. If None, a new figure object will be created.

    Methods:
    - refresh(X): Update the plot with new data X.
    """

    def __init__(self, data, plot_type, ax=None, fig=None, **kwargs):
        """
        Initialize the BlitPlot object.
        """
        self.fig = plt.figure(figsize=(6, 4)) if fig is None else fig
        self.ax = self.fig.add_subplot(111) if ax is None else ax
        self.canvas = self.fig.canvas
        self.redraw_canvas = False
        self.plot_type = plot_type

        self.canvas.draw()
        if plot_type == "line":
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.lines = self.ax.plot(data, animated=True)
        elif plot_type == "event":
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.evts = self.ax.eventplot(data, animated=True)
        elif plot_type == "image":
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.image = self.ax.imshow(data, aspect="auto", animated=True)
        elif plot_type == "trajectory":
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.lines = self.ax.plot(data[:, 0], data[:, 1], alpha=0.5, animated=True)
        elif plot_type == "trajectory3d":
            self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.lines = self.ax.plot(
                data[:, 0], data[:, 1], data[:, 2], alpha=0.5, animated=True
            )
        else:
            raise ValueError("Invalid plot type")
        ax.set(**kwargs)
        plt.show(block=False)

    def __refresh_lines(self, X):
        """
        Refresh the line plot with new data X.
        """
        # if X in 1D, convert to 2D
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
            print(X.shape)
        for i, line in enumerate(self.lines):
            line.set_ydata(X[:, i])
            self.ax.draw_artist(line)
        if np.max(X[-1, :]) > self.ax.get_ylim()[1]:
            self.ax.set_ylim(self.ax.get_ylim()[0], np.max(X[-1, :]) * 1.1)
            self.redraw_canvas = True
        if np.min(X[-1, :]) < self.ax.get_ylim()[0]:
            self.ax.set_ylim(np.min(X[-1, :]) * 1.1, self.ax.get_ylim()[1])
            self.redraw_canvas = True

    def __refresh_image(self, X):
        """
        Refresh the image plot with new data X.
        """
        self.image.set_data(X)
        self.ax.draw_artist(self.image)

    def __refresh_traj(self, X):
        """
        Refresh the 2D trajectory plot with new data X.
        """
        self.lines[0].set_data(X[:, 0], X[:, 1])
        self.ax.draw_artist(self.lines[0])

        if X[-1, 0] > self.ax.get_xlim()[1]:
            self.ax.set_xlim(self.ax.get_xlim()[0], X[-1, 0] * 1.1)
            self.redraw_canvas = True
        if X[-1, 0] < self.ax.get_xlim()[0]:
            self.ax.set_xlim(X[-1, 0] * 1.1, self.ax.get_xlim()[1])
            self.redraw_canvas = True
        if X[-1, 1] > self.ax.get_ylim()[1]:
            self.ax.set_ylim(self.ax.get_ylim()[0], X[-1, 1] * 1.1)
            self.redraw_canvas = True
        if X[-1, 1] < self.ax.get_ylim()[0]:
            self.ax.set_ylim(X[-1, 1] * 1.1, self.ax.get_ylim()[1])
            self.redraw_canvas = True

    def __refresh_traj3d(self, X):
        """
        Refresh the 2D trajectory plot with new data X.
        """
        self.lines[0].set_data(X[:, 0], X[:, 1])
        self.lines[0].set_3d_properties(X[:, 2])
        self.ax.draw_artist(self.lines[0])

        if X[-1, 0] > self.ax.get_xlim()[1]:
            self.ax.set_xlim(self.ax.get_xlim()[0], X[-1, 0] * 1.1)
            self.redraw_canvas = True
        if X[-1, 0] < self.ax.get_xlim()[0]:
            self.ax.set_xlim(X[-1, 0] * 1.1, self.ax.get_xlim()[1])
            self.redraw_canvas = True
        if X[-1, 1] > self.ax.get_ylim()[1]:
            self.ax.set_ylim(self.ax.get_ylim()[0], X[-1, 1] * 1.1)
            self.redraw_canvas = True
        if X[-1, 1] < self.ax.get_ylim()[0]:
            self.ax.set_ylim(X[-1, 1] * 1.1, self.ax.get_ylim()[1])
            self.redraw_canvas = True

    def __refesh_event(self, X, t):
        # for i, (y_val, event, spike) in enumerate(zip(y, event_spike, spikes)):
        #     if y_val > 0:
        #         spike.append(tt)
        #     if len(spike) > 0:
        #         while spike[0] < tt - t_max:
        #             spike.pop(0)
        #             if len(spike) == 0:
        #                 break
        #         event.set_positions(np.array(spike) - tt)
        #     self.ax.draw_artist(event)
        pass

    def refresh(self, X, t=None):
        """
        Refresh the plot with new data X.
        """
        self.fig.canvas.restore_region(self.bg)
        if self.plot_type == "line":
            self.__refresh_lines(X)
        elif self.plot_type == "event":
            self.__refesh_event(X, t)
        elif self.plot_type == "image":
            self.__refresh_image(X)
        elif self.plot_type == "trajectory":
            self.__refresh_traj(X)
        elif self.plot_type == "trajectory3d":
            self.__refresh_traj3d(X)
        else:
            raise ValueError("Invalid plot type")
        if self.redraw_canvas:
            self.fig.canvas.draw()
            self.redraw_canvas = False

        self.canvas.blit(self.ax.bbox)


if __name__ == "__main__":
    # Create a 2D tensor X
    X = np.random.randn(200, 10)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    test_plot = BlitPlot(X, "line", fig=fig, ax=ax)
    test_plot2 = BlitPlot(X, "line", fig=fig, ax=ax2)

    # Benchmark torch.roll along axis=0
    for i in range(1000):
        X = np.roll(X, -1, axis=0)
        test_plot.refresh(X)
        test_plot2.refresh(X)
        fig.canvas.flush_events()
