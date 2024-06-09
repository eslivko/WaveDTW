import matplotlib.pyplot as plt
from typing import List
from functions import *



class SignalModifier():
    """
    Class for creating modified version of the 1D signal based on time warping and Gaussian amplitude bump around anchor point
    """

    def __init__(self,  time_warping = False, gaussian_bump = False):
        self.time_warping_flag = time_warping
        self.gaussian_bump_flag = gaussian_bump

    def sigmoid0(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        self.sigmoid_x = self.sigmoid0(x)
        return self.sigmoid_x * (1 - self.sigmoid_x)


    def sigmoid_derivative_params(self,x, a=1.0, b=3.0):
        """
        :param x: argument of sigmoid function
        :param a: parameter that inverse proportional to the width of the sigmoid derivative "bell". default is np.pi (experimental value) - time warping window
        :param b: parameter that influence the height of the sigmoid derivative "bell", default is 1.0 - time warping rate
        :return: returns derivative of sigmoid function, adjusted based on parameters values
        """
        sigmoid_x = self.sigmoid0(x/a)
        return b * sigmoid_x * (1 - sigmoid_x) #* np.exp(-1 / (2 * a ** 2))


    def time_warping(self, x_orig, x_anchor_id, time_warping_window=np.pi, time_warping_rate=3.0):
        """
        :param x_orig: x-axis of original signal before warping
        :param x_anchor_id: id of the point that acts as the center of the synthetic warping process
        :param time_warping_window: parameter that inverse proportional to the width of the sigmoid derivative "bell". default is np.pi (experimental value)
        :param time_warping_rate: parameter that influence the height of the sigmoid derivative "bell", default is 1.0
        :return: x_warped: x-axis of the signal after warping
        """
        x_anchor = x_orig[x_anchor_id]
        dx = np.abs([x_anchor - i for i in x_orig])
        df_dx_result = self.sigmoid_derivative_params(x=dx, a=time_warping_window, b=time_warping_rate)
        x_warped = x_orig - df_dx_result
        return x_warped

    def gaussian_bumping(self, x, x_anchor_id, w=np.pi / 2, height=0.2):
        """
        Construct a Gaussian bump function that increases the amplitude
        around the anchor point x_anchor within the window w.

        Parameters:
        - x: Array of x values.
        - x_anchor: Anchor point around which the bump is centered.
        - w: Width of the window where the bump is applied.
        - height: Height of the Gaussian bump (default is 1.0).

        Returns:
        - A Gaussian bump function applied to x.
        """

        x_anchor = x[x_anchor_id]
        bump = height * np.exp(-((x - x_anchor) / (0.5 * w)) ** 2)
        return bump




    def modify(self,
               x:np.array,
               y:np.array,
               x_anchor_id:int,
               time_warping_window:float = 1.0,
               time_warping_rate:float=3.0,
               bump_window:float = np.pi / 2,
               bump_height:float = 0.2):

        self.x = x
        self.y = y
        self.x_anchor_id = x_anchor_id
        if self.time_warping_flag:
            self.x_modified = self.time_warping(x, x_anchor_id, time_warping_window=time_warping_window, time_warping_rate=time_warping_rate)
        else:
            self.x_modified = x.copy()

        if self.gaussian_bump_flag:
            self.bump = self.gaussian_bumping (x, x_anchor_id, w=bump_window, height=bump_height)
            self.y_modified = self.y.copy() + np.sign(self.y)*self.bump
        else:
            self.y_modified = y.copy()

        if self.time_warping_flag:
            self.y_interpolated = np.interp(self.x, self.x_modified, self.y_modified)

        return self.x_modified, self.y_modified, self.y_interpolated

    def modify_by_grid(self,
                       x:np.array,
                       y:np.array,
                       x_anchor_id: int,
                       time_warping_window:List,
                       time_warping_rate:List,
                       bump_window:List,
                       bump_height:List,
                       warp_and_bump = False,
                       steps=None):

        self.time_warping_flag = True
        self.gaussian_bump_flag = True
        self.warp_and_bump_flag = warp_and_bump
        self.warp_and_bump_steps = steps
        self.include_params_border_values = False


        self.x = x
        self.y = y
        self.x_anchor_id = x_anchor_id

        if self.warp_and_bump_flag and self.warp_and_bump_steps:
            self.time_warping_window = np.linspace(start=np.min(time_warping_window), stop=np.max(time_warping_window), num=self.warp_and_bump_steps)
            self.time_warping_rate = np.linspace(start=np.min(time_warping_rate), stop=np.max(time_warping_rate),
                                                   num=self.warp_and_bump_steps)
            self.bump_window = np.linspace(start=np.min(bump_window), stop=np.max(bump_window),
                                                   num=self.warp_and_bump_steps)
            self.bump_height = np.linspace(start=np.min(bump_height), stop=np.max(bump_height),
                                                   num=self.warp_and_bump_steps)

            self.include_params_border_values = True
        else:
            self.time_warping_window = time_warping_window
            self.time_warping_rate = time_warping_rate
            self.bump_window = np.rand(bump_window,2)
            self.bump_height = bump_height


        #param_dict - dictionary containing all values for parameters
        param_dict = {
                "time_warping_window":self.time_warping_window,
                "time_warping_rate":self.time_warping_rate,
                "bump_window":self.bump_window,
                "bump_height":self.bump_height
                          }
        #create parameter grid as a List, containing dictionaries with applicable combinations of parameters
        self.result_grid = dict_configs(param_dict = param_dict,
                                        increase_params_simultaneously_flag = self.warp_and_bump_flag,
                                        include_params_border_values_flag= self.include_params_border_values)


        for i, param_config in enumerate(self.result_grid):
            x_modified = self.time_warping(x_orig=self.x,
                                           x_anchor_id = self.x_anchor_id,
                                           time_warping_window=param_config["time_warping_window"],
                                           time_warping_rate=param_config["time_warping_rate"]
                                           )

            bump = self.gaussian_bumping (x=self.x,
                                          x_anchor_id = self.x_anchor_id,
                                          w=param_config["bump_window"],
                                          height=param_config["bump_height"]
                                          )
            y_modified = y.copy() + bump
            if self.time_warping_flag:
                y_interpolated = np.interp(self.x, x_modified, y_modified)

            self.result_grid[i]["x_modified"] = x_modified
            self.result_grid[i]["y_modified"] = y_modified
            self.result_grid[i]["y_interpolated"] = y_interpolated


        return self.result_grid



    def plot_modified(self):
        # Interpolate missing data points

        #self.resampled = resample(self.y_modified,len(self.y))
        #self.interpolated = np.interp(self.x,self.x_modified, self.y_modified)

        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.y, c='#4682B4',lw = 3,label='Original Sine Wave')
        #plt.plot(self.x_modified, self.y_modified, label='Modified Sine Wave')
        plt.plot(self.x, self.y_modified, lw = 3,c='#3CB371',label='Modified Sine Wave')


        #plt.scatter(self.x[self.x_anchor_id],0 , color = 'red')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Sine Wave with {"Time warping, " if self.time_warping_flag else "no Time warping,"} {"Gaussian Amplitude Bump"if self.gaussian_bump_flag else "no Gaussian Amplitude Bump"}')
        plt.grid(True)
        plt.show()


    def plot_modified_grid(self,savefig = False, filepath=None):
        plt.figure(figsize=(10, 5))
        x_peaks = find_signal_peaks(self.y)
        plt.scatter(self.x[x_peaks], self.y[x_peaks], s = 12, c = 'r', zorder=2)

        grid_size = len(self.result_grid)
        for i, result_wave in enumerate(self.result_grid):
            label = r"$Q^{\prime}$," \
                    fr"b"\
                    fr"={np.round(result_wave['time_warping_rate'],2)}," \
                    fr"h" \
                    fr"={np.round(result_wave['bump_height'],2)}"
                    #f"w={np.round(result_wave['time_warping_window'],2)}," \
                    #f"w={np.round(result_wave['bump_window'],2)}," \

            color_step = np.round(0.4/self.warp_and_bump_steps,2)
            np.random.seed(34187 + i*200)
            if ((i) % (self.warp_and_bump_steps-1)) == 0 or i==0:
                r = np.round(np.random.uniform(0,0.5) , 1)
                g = np.round(np.random.uniform(0,0.5) , 1)
                b = np.round(np.random.uniform(0, 0.5), 1)
                #b = np.round(np.random.rand(), 1)

            r = r + color_step
            g = g + color_step
            b = b + color_step

            c = [r, g, b]

            plt.plot(result_wave["x_modified"], result_wave["y_modified"], alpha = 0.9,  color=c, label=label)
        plt.plot(self.x, self.y, label=r'Original signal $Q$', linewidth=3)

        plt.legend(loc='upper right',fontsize= 'small')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(f'Signal companion generation with {"time warping, " if self.time_warping_flag else "no time warping,"} {"Gaussian amplitude bump"if self.gaussian_bump_flag else "no Gaussian Amplitude Bump"}',fontsize= 'medium')
        plt.grid(True)


        if savefig == True:
            plt.savefig(filepath, bbox_inches='tight')
        plt.show()


