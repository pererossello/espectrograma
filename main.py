import os
import copy
import datetime
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import scipy as sp
from pydub import AudioSegment

import utils as ut
import plot_utils as pu


class Spectrogram():

    def __init__(self, path_audio):

        self.path_audio = path_audio

        format = path_audio.split('.')[-1]

        audio = AudioSegment.from_file(path_audio, format=format)  # audio data
        self.sf = audio.frame_rate  # frame rate

        # convert to mono
        self.audio_data = np.mean(np.array(audio.get_array_of_samples()).reshape((-1, 2)), axis=1)  

        # keep he original audio data
        self.audio_data_raw = copy.deepcopy(self.audio_data)

        self.dt = 1.0 / self.sf  # Time step

        # frequency range default
        self.fmin = ut.chromatic_scale['C1']
        self.fmax = ut.chromatic_scale['C8']

        self.plot_params_set = False
        self.title_set = False
        self.xticks_set = False
        self.yticks_set = False
        self.modify_x = False
        self.modify_y = False
        self.modify_t = False


    def set_freq_range(self, fmin, fmax):

        # if fmin and fmax are string:
        if isinstance(fmin, str):
            fmin = ut.chromatic_scale[fmin]
        if isinstance(fmax, str):
            fmax = ut.chromatic_scale[fmax]

        self.fmin = fmin
        self.fmax = fmax

    def trim_audio(self, tmin, tmax):

        self.tmin = tmin
        self.tmax = tmax

        self.audio_data = self.audio_data_raw[int(tmin / self.dt) : int(tmax / self.dt)]


    def power_spectrum(self, audio_data):
        result = (2.0 / (self.N // 2)) * np.abs(sp.fft.fft(audio_data)[: self.N // 2])[(self.f>self.fmin) & (self.f<self.fmax)]
        return result


    def make_spectrogram(self, bin_size, frequency_resolution=200, moving_average=True):

        self.bin_size = bin_size
        N = int(self.bin_size // self.dt)  # Number of data points per bin 
        M = int(len(self.audio_data) // N)  # Number of bins in the spectrogram

        self.N = N
        self.M = M

        audio_data_clip = self.audio_data[:int(M * N)]
        self.audio_data_arr = audio_data_clip.reshape((M, N))

        self.f = sp.fft.fftfreq(N, self.dt)[: N // 2]
        f_trimmed = self.f[(self.f>self.fmin) & (self.f<self.fmax)]

        ff = [np.linspace(0, 1, frequency_resolution)]  # Start with the initial linspace
        ff += [np.linspace(2**i, 2**(i+1), frequency_resolution) for i in range(11)]  # Append other ranges
        ff = np.concatenate(ff)
        ff = ff[(ff>self.fmin) & (ff<self.fmax)]

        def interpolate(f):
            result = np.interp(ff, f_trimmed, f)
            return result
        
        self.spec = np.apply_along_axis(self.power_spectrum, 1, self.audio_data_arr)
        self.spec = np.apply_along_axis(interpolate, 1, self.spec)

        self.ff = ff
        self.frequency_resolution = frequency_resolution

        if moving_average is not False:

            if moving_average is True:
                self.moving_average()
            else:
                self.moving_average(win=moving_average)


    def moving_average(self, win=None):

        if win is None:
            win = self.frequency_resolution // 20

        def moving_average(a, n=win):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1 :] / n
        
        self.spec = np.apply_along_axis(moving_average, 1, self.spec)
        self.ff = self.ff[win // 2 - 1 : -win // 2]

    def set_title(self, title='', position=(0.015,0.945), color=None, alpha=1, fontsize=1.25, modify='l0.4'):

        if self.plot_params_set == False:
            self.set_plot_params()

        self.title = title
        self.title_position = position
        if color is None:
            color = self.cmap_1(0)
        self.title_color = color
        self.title_alpha = alpha
        self.title_fontsize = fontsize

        self.title_set = True

        if modify is not None:
            
            self.modify_t_type = modify[0]
            self.modify_t_value = np.float32(modify[1:])

            # raise error if modify type is not 'l' or 'd':
            if self.modify_t_type not in ['l', 'd']:
                raise ValueError('modify parameter must be either "l{float}" or "d{float}" where float is between 0 and 1')
            
            self.modify_t = True

    def set_xticks(self, label_pad=1.2, label_size=1.25, alpha=1, color=None, modify='l0.4', key='C_major'):

        if self.plot_params_set == False:
            self.set_plot_params()

        self.x_ticks =  {k: ut.chromatic_scale[k] for k in ut.scales[key] if k in ut.chromatic_scale} 

        self.xtick_label_pad = label_pad
        self.xtick_label_size = label_size
        self.xtick_label_alpha = alpha
        if color is None:
            color = self.cmap_1(0)
        self.xtick_color = color

        self.xticks_set = True

        if modify is not None:
            
            self.modify_x_type = modify[0]
            self.modify_x_value = np.float32(modify[1:])

            # raise error if modify type is not 'l' or 'd':
            if self.modify_x_type not in ['l', 'd']:
                raise ValueError('modify parameter must be either "l{float}" or "d{float}" where float is between 0 and 1')
            
            self.modify_x = True

        return
    
    def set_yticks(self, label_pad=1, label_size=1.1, alpha=1, color=None, jump=7, modify='l0.4'):

        if self.plot_params_set == False:
            self.set_plot_params()

        self.ytick_label_pad = label_pad
        self.ytick_label_size = label_size
        self.ytick_label_alpha = alpha
        if color is None:
            color = self.cmap_1(0)
        self.ytick_color = color

        self.jump = jump

        self.yticks_set = True

        if modify is not None:
            
            self.modify_y_type = modify[0]
            self.modify_y_value = np.float32(modify[1:])

            # raise error if modify type is not 'l' or 'd':
            if self.modify_y_type not in ['l', 'd']:
                raise ValueError('modify parameter must be either "l{float}" or "d{float}" where float is between 0 and 1')
            
            self.modify_y = True

        return


    def set_plot_params(self, fig_size=1080, dpi=500, ratio=3/2, cmap1=('bone', 0.2, 1), cmap2=('jet', 0, 1),
                        color_scale=0.75, number_lines=100, facecolor='k', scale=1, linewidth=0.075, y_max=1.1,
                        color_spectra_power=1.5):

        self.spec_norm = self.spec / np.max(self.spec)

        self.fig_size = fig_size
        self.dpi = dpi
        self.ratio = ratio

        if isinstance(cmap1, str):
            self.cmap_1 = plt.get_cmap(cmap1)
        elif isinstance(cmap1, tuple):
            cmap_1 = plt.get_cmap(cmap1[0])
            colors = cmap_1(np.linspace(cmap1[1],cmap1[2], 256))
            self.cmap_1 = LinearSegmentedColormap.from_list('truncated_bone', colors)
        else:
            self.cmap_1 = cmap1

        if isinstance(cmap2, str):
            self.cmap_2 = plt.get_cmap(cmap2)
        elif isinstance(cmap2, tuple):
            cmap_2 = plt.get_cmap(cmap2[0])
            colors = cmap_2(np.linspace(cmap2[1],cmap2[2], 256))
            self.cmap_2 = LinearSegmentedColormap.from_list('truncated_bone', colors)
        else:
            self.cmap_2 = cmap2

        self.norm_1 = plt.Normalize(self.spec_norm.min(), self.spec_norm.max()*color_scale)
        self.norm_2 = plt.Normalize(self.fmin, self.fmax)

        self.L = number_lines

        self.facecolor = facecolor

        self.scale = scale*0.15
        self.lw = linewidth

        self.y_max = y_max

        self.color_spectra_power = color_spectra_power

        self.plot_params_set = True


    def plot(self, show=False, savepath=None, bbox_inches='tight'):

        if not self.plot_params_set:
            self.set_plot_params()

        Fig = pu.Figure(fig_size=self.fig_size, ratio=self.ratio, dpi=self.dpi, facecolor=self.facecolor)
        ax = Fig.ax
        fs = Fig.fs
        self.fig = Fig.fig

        spec_plot = self.spec_norm * self.scale
        self.spec_plot = spec_plot

        steps = [int(val) for val in np.linspace(0, self.M-1, self.L)]

        def sigmoid(x):
            return 1*(1 / (1 + np.exp(20*(0.4-x))))

        secs = []
        for i, step in enumerate(steps):
            y_0 = 1-i/len(steps)
            secs.append([y_0, step*self.bin_size])

            ax.fill_between(self.ff, spec_plot[step,:]+y_0, np.full((len(self.ff)), y_0), facecolor=self.facecolor, linewidth=0.0, zorder=i-1)
            
            points = np.array([self.ff, spec_plot[step, :] + y_0]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments,
                                cmap=self.cmap_1,
                                norm=self.norm_1,
                                alpha=1,
                                linewidth=self.lw*fs*2,
                                zorder=i)
            
            lc.set_array(self.spec_norm[step, :])
            line1 = ax.add_collection(lc)

            alphas =  (self.spec_norm[step,:])/self.spec_norm.max()
            #alphas = np.clip(alphas, 0, 1)
            alphas = alphas**(1/self.color_spectra_power)
            alphas = sigmoid(alphas)

            lc_2 = LineCollection(segments,
                        cmap=self.cmap_2,
                        norm=self.norm_2,
                        alpha= alphas,
                        linewidth=self.lw*fs*2,
                        zorder=i)

            lc_2.set_array(self.ff)
            line2 = ax.add_collection(lc_2)

        self.secs = secs

        ax.set_ylim(0,self.y_max)
        ax.set_xscale("log", base=2)


        # Setting X ticks

        def superscript(note):
            if '#' in note:
                new_note = note[0] + '$^{{\#}}_{}$'.format(note[-1])
                return new_note
            else:
                new_note = note[0] + '$_{}$'.format(note[-1])
                return new_note

        if self.xticks_set == False:
            self.set_xticks()

        ax.set_xticks(list(self.x_ticks.values())) 

        new_labels = [superscript(i) for i in list(self.x_ticks.keys())]

        ax.set_xticklabels(new_labels,
                        fontsize=self.xtick_label_size*fs,
                        alpha=self.xtick_label_alpha, 
                        va='center',
                        ha='center',
                        color=self.xtick_color)
        
        ax.set_xlim(np.min(self.ff), np.max(self.ff))
        for label in ax.get_xticklabels():
            dif = self.fmax - self.fmin
            position = label.get_position()
            rgb = (position[0] - self.fmin)/dif
            c = list(self.cmap_2(rgb))
            if self.modify_x == True:
                if self.modify_x_type == 'l':
                    c = ut.lighten_color(c, amount=self.modify_x_value)
                elif self.modify_x_type == 'd':
                    c = ut.darken_color(c, amount=self.modify_x_value)
            label.set_color(tuple(c))

        ax.tick_params(axis='x', which='major', pad=self.xtick_label_pad*fs, size=0)


        # Setting Y ticks

        if self.yticks_set == False:
            self.set_yticks()

        if self.modify_y == True:
            if self.modify_y_type == 'l':
                c_y = ut.lighten_color(self.ytick_color, amount=self.modify_y_value)
            elif self.modify_y_type == 'd':
                c_y = ut.darken_color(self.ytick_color, amount=self.modify_y_value)
        else:
            c_y = self.ytick_color

        secs = np.array(secs)
        y_labels = [str(datetime.timedelta(seconds=i))[2:7] for i in secs[:,1]][::self.jump]
        ax.set_yticks(secs[:,0][::self.jump])
        ax.set_yticklabels(y_labels,
                        fontsize=self.ytick_label_size*fs,
                        alpha=self.ytick_label_alpha,
                        color=c_y,
                        )
        ax.yaxis.tick_right()

        ax.tick_params(axis='y', which='major', pad=self.ytick_label_pad*fs, size=0)
        

        # Setting title

        if self.title_set == False:
            self.set_title()
        
        if self.modify_t == True:
            if self.modify_t_type == 'l':
                c_t = ut.lighten_color(self.title_color, amount=self.modify_t_value)
            elif self.modify_t_type == 'd':
                c_t = ut.darken_color(self.title_color, amount=self.modify_t_value)
        else:
            c_t = self.title_color

        ax.set_title(self.title,
                    fontsize=self.title_fontsize*fs,
                    alpha=self.title_alpha,
                    color=c_t,
                    va='center',
                    ha='left',
                    y=self.title_position[1],
                    x=self.title_position[0])
        

        if show:
            plt.show()
        else:
            plt.close()

        if savepath is not None:

            if not bbox_inches == 'tight':

                fh = Fig.fig_height
                fw = Fig.fig_width

                a, b, c, ratio = bbox_inches

                dr_x = c
                x0, y0 = a*fw, b*fw*ratio

                x0, x1 = x0, x0 + fw*(dr_x)
                y0, y1 = y0, y0 + fw*(dr_x)*ratio

                Fig.save(savepath, bbox_inches=mpl.transforms.Bbox([[x0, y0], [x1, y1]])                 
                        )
                
            else:

                Fig.save(savepath, bbox_inches=bbox_inches                  
                        )

        self.Fig = Fig

        self.plot_params_set = False
        self.title_set = False
        self.xticks_set = False
        self.yticks_set = False
        self.modify_x = False
        self.modify_y = False
        self.modify_t = False


    def check_saved_image(self):

        self.Fig.check_saved_image()



    def generate_frames(self, savefolder='../figures/', filename='image', bbox_inches='tight', extension='.jpg', animation_frames=5):

        self.savefolder = savefolder
        self.animation_frames = animation_frames

        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        else:
            shutil.rmtree(savefolder)
            os.makedirs(savefolder)

        if not self.plot_params_set:
            self.set_plot_params()

        Fig = pu.Figure(fig_size=self.fig_size, ratio=self.ratio, dpi=self.dpi, show=False, facecolor=self.facecolor)
        ax = Fig.ax
        fs = Fig.fs
        self.fig = Fig.fig

        spec_plot = self.spec_norm * self.scale
        self.spec_plot = spec_plot

        steps = [int(val) for val in np.linspace(0, self.M-1, self.L)]

        def sigmoid(x):
            return 1*(1 / (1 + np.exp(20*(0.4-x))))

        ax.set_ylim(0,self.y_max)
        ax.set_xscale("log", base=2)


        # Setting X ticks

        def superscript(note):
            
            if '#' in note:
                new_note = note[0] + '$^{{\#}}_{}$'.format(note[-1])
                return new_note
            else:
                new_note = note[0] + '$_{}$'.format(note[-1])
                return new_note
    


        if self.xticks_set == False:
            self.set_xticks()

        ax.set_xticks(list(self.x_ticks.values())) 
        new_labels = [superscript(i) for i in list(self.x_ticks.keys())]

        ax.set_xticklabels(new_labels,
                        fontsize=self.xtick_label_size*fs,
                        alpha=self.xtick_label_alpha, 
                        va='center',
                        ha='center',
                        color=self.xtick_color)
        
        ax.set_xlim(self.fmin, self.fmax)
        for label in ax.get_xticklabels():
            dif = self.fmax - self.fmin
            position = label.get_position()
            rgb = (position[0] - self.fmin)/dif
            c = list(self.cmap_2(rgb))
            if self.modify_x == True:
                if self.modify_x_type == 'l':
                    c = ut.lighten_color(c, amount=self.modify_x_value)
                elif self.modify_x_type == 'd':
                    c = ut.darken_color(c, amount=self.modify_x_value)
            label.set_color(tuple(c))

        ax.tick_params(axis='x', which='major', pad=self.xtick_label_pad*fs, size=0)


        # Setting Y ticks

        secs = []
        for i, step in enumerate(steps):
            y_0 = 1-i/len(steps)
            secs.append([y_0, step*self.bin_size])

        if self.yticks_set == False:
            self.set_yticks()

        if self.modify_y == True:
            if self.modify_y_type == 'l':
                c_y = ut.lighten_color(self.ytick_color, amount=self.modify_y_value)
            elif self.modify_y_type == 'd':
                c_y = ut.darken_color(self.ytick_color, amount=self.modify_y_value)
        else:
            c_y = self.ytick_color

        secs = np.array(secs)
        y_labels = [str(datetime.timedelta(seconds=i))[2:7] for i in secs[:,1]][::self.jump]
        ax.set_yticks(secs[:,0][::self.jump])
        ax.set_yticklabels(y_labels,
                        fontsize=self.ytick_label_size*fs,
                        alpha=self.ytick_label_alpha,
                        color=c_y,
                        )
        ax.yaxis.tick_right()

        ax.tick_params(axis='y', which='major', pad=self.ytick_label_pad*fs, size=0)
        

        # Setting title

        if self.title_set == False:
            self.set_title()
        
        if self.modify_t == True:
            if self.modify_t_type == 'l':
                c_t = ut.lighten_color(self.title_color, amount=self.modify_t_value)
            elif self.modify_t_type == 'd':
                c_t = ut.darken_color(self.title_color, amount=self.modify_t_value)
        else:
            c_t = self.title_color

        ax.set_title(self.title,
                    fontsize=self.title_fontsize*fs,
                    alpha=self.title_alpha,
                    color=c_t,
                    va='center',
                    ha='left',
                    y=self.title_position[1],
                    x=self.title_position[0])


        J = animation_frames


        k = 0
        for i, step in enumerate(steps):
            y_0 = 1-i/len(steps)
            
            for j, fact in enumerate(np.linspace(0, 1, J)):

                alpha_ = 0.1 + 2*j/(J-1)
                # clip to 1
                alpha_ = min(alpha_, 1)
                fill = ax.fill_between(self.ff, fact*spec_plot[step,:]+y_0, np.full((len(self.ff)), y_0), facecolor=self.facecolor, linewidth=0.0, zorder=2*i-1)
                
                points = np.array([self.ff, fact*spec_plot[step, :] + y_0]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(segments,
                                    cmap=self.cmap_1,
                                    norm=self.norm_1,
                                    alpha=alpha_,
                                    linewidth=self.lw*fs*2,
                                    zorder=2*i)
                
                lc.set_array(self.spec_norm[step, :])
                line1 = ax.add_collection(lc)

                alphas =  (self.spec_norm[step,:])/self.spec_norm.max()
                #alphas = np.clip(alphas, 0, 1)
                alphas = alphas**(1/self.color_spectra_power)*alpha_
                alphas = sigmoid(alphas)

                lc_2 = LineCollection(segments,
                            cmap=self.cmap_2,
                            norm=self.norm_2,
                            alpha= alphas,
                            linewidth=self.lw*fs*2,
                            zorder=2*i+1)

                lc_2.set_array(self.ff)
                line2 = ax.add_collection(lc_2)

                savepath = savefolder + filename + f'_{k:04d}{extension}'

                Fig.save(savepath, bbox_inches=bbox_inches                  
                        )
                
                if j != J-1:
                    line1.remove()
                    line2.remove()
                    fill.remove()
                
                k += 1

        self.Fig = Fig

        self.plot_params_set = False
        self.title_set = False
        self.xticks_set = False
        self.yticks_set = False
        self.modify_x = False
        self.modify_y = False
        self.modify_t = False


    def frames_to_video(self, audio_delay=10):
        audio_duration = len(self.audio_data) * self.dt
        total_frames = self.animation_frames * self.L
        fps = total_frames/(audio_duration)

        pu.png_to_mp4(self.savefolder, extension='.jpg', fps=fps, title='video')

        path_video = os.path.dirname(os.path.dirname(self.savefolder)) + '/video.mp4'
        path_output = os.path.dirname(os.path.dirname(self.savefolder)) + '/video_with_audio.mp4'

        pu.add_audio_to_video(path_video, self.path_audio, path_output, audio_delay_ms=audio_delay)
