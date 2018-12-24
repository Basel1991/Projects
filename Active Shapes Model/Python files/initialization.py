"""
Created on Sun Nov  11 17:17:25 2018
This class is used to initialize easily the Active Shape Models
@author: Basel Alyafi
"""
import numpy as np

from align import (align, transform)


class DynamicShape:

    def __init__(self, shape, figure, plot, param):
        """
        a constructor for the DynamicShape class
        :param shape: 2n float vector
            the first n values correspond to the x coordinates of the shape, the next n values correspond to the
            y coordinates
        :param figure: a figure handle
        :param plot: a plot handle
        :param dict: a dictionary containing scale, translate_x, translate_y, theta
        """
        self.shape = shape
        self.right_pressed = False
        self.left_pressed = False
        self.scale = param["scale"]
        self.translate_x = param["translate_x"]
        self.translate_y = param["translate_y"]
        self.theta = param["theta"]
        self.figure = figure
        self.plot = plot

    def connect(self):
        'connect to all events needed'
        self.figure.canvas.mpl_connect('button_press_event',  self.on_click)
        self.figure.canvas.mpl_connect('scroll_event', self.wheel_event)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_click(self, event):
        """
        left and right mouse click event listener
        :param event: the event
        :return: void
        """
        # global scale
        # global translate_x
        # global translate_y
        # print('x ', event.x, ' y ', event.y, ' x data ', event.xdata, ' y data ', event.ydata)
        if event.button == 1:
            self.left_pressed = True
            self.translate_x = event.xdata
            self.translate_y = event.ydata
        elif event.button == 3:
            self.right_pressed = True
        self.draw()
        # fig.canvas.flush_events()

        print('Tx %1.4f Ty %1.4f  ' %(self.translate_x, self.translate_y))

    def wheel_event(self, event):
        """
        a listener for the wheel movement, forward movement to enlarge the shape, backward movement to shrink it.
        :param event: the event of wheel movement
        :return: void
        """
        if event.button == 'up':
            self.scale = self.scale + 50
        elif event.button == 'down':
            self.scale = self.scale - 50
        print('scale ',self.scale)
        self.draw()

    def draw(self):
        """
        to draw the new shape by updating the x and y variables
        :return: void
        """
        aligned_shape = transform(self.shape, self.scale, self.theta, self.translate_x, self.translate_y)
        num_points = int(self.shape.__len__()/2)
        self.plot.set_ydata(aligned_shape[num_points:])
        self.plot.set_xdata(aligned_shape[:num_points])
        self.figure.suptitle('scale %1.4f, theta %1.4f, translate_x %1.4f, translate_y %1.4f'%(self.scale, self.theta, self.translate_x, self.translate_y))
        self.figure.canvas.draw()

    def on_motion(self, event):
        """
        a listener for the movement of the mouse over the figure
        :param event: the event of mouse movement
        :return: void
        """
        if (event.button == 1) & self.left_pressed:
            self.translate_x = event.xdata
            self.translate_y = event.ydata
            print('Tx %1.4f Ty %1.4f  ' % (self.translate_x, self.translate_y))
        elif (event.button == 3) & self.right_pressed:
            angle = event.xdata - self.translate_x
            self.theta = angle/300
            print('angle ',self.theta)

        self.draw()

    def on_release(self, event):
        """
        listens to the release of mouse buttons
        :param event: the event of releasing mouse buttons
        :return: void
        """
        if event.button == 1:
            self.left_pressed = False
        elif event.button == 3:
            self.right_pressed = False

    def get_parameters(self):
        """
        to return the parameters needed to reproduce the shape
        :return: scale, angle, translation over x axis, translation over y axis
        """
        return self.scale, self.theta, self.translate_x, self.translate_y


# print('after plotting')

