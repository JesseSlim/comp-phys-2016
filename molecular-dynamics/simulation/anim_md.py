import numpy as np
#plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import juggle_axes
import matplotlib.colors as colors
import matplotlib.cm as cm

class AnimatedScatter(object):
    def __init__(self, numpoints, box_len, temp, pos, mom, part_update, *args):
        """
        Class for particle animation
        constructor takes as arguments                                                              
        -- numpoints: the number of points                                                        
        -- box_len: the side of the (cubic) box                                                   
        -- pos: numpy array of shape [numpoints, 3] containing the position coordinates           
        -- mom: numpy array of shape [numpoints, 3] containing the momentum coordinates
        -- part_update: function which calculates the new particle positions
           This function must output the updated arrays pos and mom                      
        -- args: the function part_update takes as arguments numpoints, box_len, pos, mom, *args 
        Example: anim_md.AnimatedScatter(n, box_len, pos, mom, part_update, n_t, dt)
        where 'part_update' is defined as
        def part_update(n, box_len, temp, pos, mom, n_t, dt):
          ...
          ...
          return pos, mom
        Note: temp is the temperature, n_t is the number of iterations in part_update and dt is the timestep
        """
        self.numpoints = numpoints
        self.pos = pos
        self.mom = mom
        self.box_len = box_len
        self.arglist = args
        self.temp = temp
        
        self.stream = self.data_stream()
        self.angle = 30
        self.part_update = part_update
        self.fig, self.ax = plt.subplots(figsize=(13,9.5))
        self.FLOOR = 0.0
        self.CEILING = self.box_len
        self.ax = self.fig.add_subplot(111,projection = '3d')
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, 
                                           init_func=self.setup_plot, blit=True)
#        self.trace = 
        
    def change_angle(self):
        """ Change angle for each rotation step """
        self.angle = (self.angle + 2)%360

    def setup_plot(self):
        """ Set world coordinates, colors, symbol size ('s') """
        x, y, z = next(self.stream)
        color_nums = np.ndarray((self.numpoints),dtype=float)
        color_nums[:] = 0.0
#        color_nums[10] = 1.0
        colors = cm.Blues(color_nums)
        sizes = np.zeros(self.numpoints)
        sizes[:] = 3
#        sizes[10] = 100
        self.scat = self.ax.scatter(x, y, z, c=colors, s=sizes, animated=True)
#        self.line = self.ax.plot(self.trace[:][0], self.trace[:][1])
        self.ax.set_xlim(self.FLOOR, self.CEILING)
        self.ax.set_ylim(self.FLOOR, self.CEILING)
        self.ax.set_zlim(self.FLOOR, self.CEILING)

        return self.scat, # self.line,

    def dens_update(self,density):
        old_dens = self.numpoints/self.box_len**3
        scale = (old_dens/density)**0.3333333333
        self.box_len = scale*self.box_len
        self.pos = self.pos*scale
        self.FLOOR = 0.0
        self.CEILING = self.box_len
        self.ax.set_xlim3d(self.FLOOR, self.CEILING)
        self.ax.set_ylim3d(self.FLOOR, self.CEILING)
        self.ax.set_zlim3d(self.FLOOR, self.CEILING)

    
    def temp_update(self,temp):
        self.temp = temp
#        print ("hoho", old_dens, density, scale, self.box_len)

    def data_stream(self):
        """ 
           Calls particle update routine, copies it to the relevant section of the 'data' array which 
           is then yielded
        """
        self.pos, self.outlist = self.part_update(self.numpoints, self.box_len, self.temp, self.pos, self.mom, *self.arglist)
        data = np.transpose(self.pos)
        while True:
            self.pos, self.mom = self.part_update(self.numpoints, self.box_len, self.temp, self.pos, self.mom, *self.arglist)
            data[:3, :] = np.transpose(self.pos)
            yield data

    def update(self, i):
        """ Use new particle position for drawing next frame """
        data = next(self.stream)
        data = np.transpose(data)

#        self.scat._offsets3d = juggle_axes(data[:,0],data[:,1],data[:,2], 'z')
#        self.ax.cla()
        self.change_angle()
        self.ax.view_init(30,self.angle)
#        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        plt.draw()
        return self.scat,


    def show(self):
#        plt.subplots_adjust(left=0.0, right=1.0, top=1.00, bottom=0.0)
        plt.show()

