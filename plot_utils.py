import os
import subprocess

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import PIL


class Figure:
    def __init__(self, fig_size=540, ratio=16/9, dpi=300,
                 text_color='w',
                 facecolor = 'k',
                 ts=2, pad=0.2, 
                 show=True):

        fig_width, fig_height = fig_size / dpi, fig_size * ratio / dpi
        fs = np.sqrt(fig_width * fig_height)

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, layout='none')

        fig.patch.set_facecolor(facecolor)


        ax = fig.add_subplot(111)
        ax.set_facecolor(facecolor)

        # no xticks nor yticks
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0)
            # spine.set_color('r')

        self.fig = fig
        self.dpi = dpi
        self.ax = ax
        self.fs = fs
        self.ratio = ratio
        self.fig_width = fig_width
        self.fig_height = fig_height

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, path, bbox_inches=None, pad_inches=None):

        self.fig.savefig(path, dpi=self.dpi, bbox_inches=bbox_inches, pad_inches=pad_inches)

        self.path = path

    def check_saved_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')


        with Image.open(self.path) as img:
            print('Image dimensions', img.size)
            print('y/x ratio:',img.size[1] / img.size[0])
            return
        
    def show_image(self):

        if not hasattr(self, 'path'):
            raise ValueError('Figure has not been saved yet.')
        
        with Image.open(self.path) as img:
            img.show()
            return
        




def png_to_mp4(fold, title='video', fps=36, digit_format='04d', res=None, resize_factor=1, custom_bitrate=None, extension='.jpg'):

    # Get a list of all .png files in the directory
    files = [f for f in os.listdir(fold) if f.endswith(extension)]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not files:
        raise ValueError("No PNG files found in the specified folder.")

    im = PIL.Image.open(os.path.join(fold, files[0]))
    resx, resy = im.size

    if res is not None:
        resx, resy = res
    else:
        resx = int(resize_factor * resx)
        resy = int(resize_factor * resy)
        resx += resx % 2  # Ensuring even dimensions
        resy += resy % 2

    basename = os.path.splitext(files[0])[0].split('_')[0]

    ffmpeg_path = 'ffmpeg'
    abs_path = os.path.abspath(fold)
    parent_folder = os.path.dirname(abs_path) + os.sep
    output_file = os.path.join(parent_folder, f"{title}.mp4")
    
    crf = 2  # Lower for higher quality, higher for lower quality
    bitrate = custom_bitrate if custom_bitrate else "5000k"
    preset = "slow"
    tune = "film"

    command = f'{ffmpeg_path} -y -r {fps} -i {os.path.join(fold, f"{basename}_%{digit_format}{extension}")} -c:v libx264 -profile:v high -crf {crf} -preset {preset} -tune {tune} -b:v {bitrate} -pix_fmt yuv420p -vf scale={resx}:{resy} {output_file}'


    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error during video conversion:", e)




def add_audio_to_video(video_file, audio_file, output_file, audio_delay_ms=0, audio_volume=1.0):
    """
    Adds an MP3 audio track to an MP4 video file with a delay.

    :param video_file: Path to the video file (MP4).
    :param audio_file: Path to the audio file (MP3).
    :param output_file: Path for the output file.
    :param audio_delay_ms: Delay for the audio track in milliseconds.
    :param audio_volume: Volume of the audio (1.0 for original volume).
    """
    # if output file exists, delete it
    if os.path.isfile(output_file):
        os.remove(output_file)


    command = [
        'ffmpeg',
        '-i', video_file,
        '-i', audio_file,
        '-filter_complex', f"[1:a]adelay={audio_delay_ms}|{audio_delay_ms},volume={audio_volume}[a]",
        '-map', '0:v', '-map', '[a]',
        '-c:v', 'copy', '-shortest',
        output_file
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Video with audio added successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")