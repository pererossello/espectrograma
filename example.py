import plot_utils as pu

name = 'bach_a_minor'

savefolder= f'figures/{name}/pngs'

fps = 1*5*194/(57.9)
print(fps)

#pu.png_to_mp4(savefolder, extension='.jpg', fps=fps)

path_audio = f'songs/{name}.mp3'
path_video = f'figures/{name}/video.mp4'
path_output = f'figures/{name}/video_with_audio.mp4'

pu.add_audio_to_video(path_video, path_audio, path_output, audio_delay_ms=10)