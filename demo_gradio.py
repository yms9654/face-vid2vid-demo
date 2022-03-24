import os
import yaml

import imageio, cv2
from moviepy.editor import *
from skimage.transform import resize
from skimage import img_as_ubyte

from demo import load_checkpoints, make_animation, find_best_frame
import gradio as gr

config = 'config/vox-256-spade.yaml'
checkpoint = 'ckpt/00000189-checkpoint.pth.tar'

gen = 'spade'
cpu = False

generator, kp_detector, he_estimator = load_checkpoints(config_path=config, checkpoint_path=checkpoint, gen=gen, cpu=cpu)


def inference(source,
              driving,
              find_best_frame_ = False,
              free_view = False,
              yaw = None,
              pitch = None,
              roll = None,
              output_name = 'output.mp4',
              
              audio = True,
              cpu = False,
              best_frame = None,
              relative = True,
              adapt_scale = True,
              ):

    # source 
    source_image = resize(source, (256, 256))
    
    # driving
    reader = imageio.get_reader(driving)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    
    with open(config) as f:
        config_ = yaml.load(f)
    estimate_jacobian = config_['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    if find_best_frame_ or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
    
    # save video
    output_path = 'asset/output'
    os.makedirs(output_path, exist_ok=True)
    
    print(f'{output_path}/{output_name}')    
    
    imageio.mimsave(f'{output_path}/{output_name}', [img_as_ubyte(frame) for frame in predictions], fps=fps)
    
    if audio:
        audioclip = VideoFileClip(driving)
        audio = audioclip.audio
        videoclip = VideoFileClip(f'{output_path}/{output_name}')
        videoclip.audio = audio
        name = output_name.strip('.mp4')
        videoclip.write_videofile(f'{output_path}/{name}_audio.mp4')
        return f'./{output_path}/{name}_audio.mp4'
    else:
        return f'{output_path}/{output_name}'
    
    
import gradio as gr

samples = []

example_source = os.listdir('asset/source')
for image in example_source:
    samples.append([f'asset/source/{image}', None])

example_driving = os.listdir('asset/driving')
for video in example_driving:
    samples.append([None, f'asset/driving/{video}'])

iface = gr.Interface(
    inference, # main function
    inputs = [ 
        gr.inputs.Image(shape=(255, 255), label='Source Image'), # source image
        gr.inputs.Video(label='Driving Video', type='mp4'), # driving video
        
        gr.inputs.Checkbox(label="fine best frame", default=False), 
        gr.inputs.Checkbox(label="free view", default=False), 
        gr.inputs.Slider(minimum=-90, maximum=90, default=0, step=1, label="yaw"),
        gr.inputs.Slider(minimum=-90, maximum=90, default=0, step=1, label="pitch"),
        gr.inputs.Slider(minimum=-90, maximum=90, default=0, step=1, label="raw"),
        
    ],
    outputs = [
        gr.outputs.Video(label='result') # generated video
    ], 
    
    title = 'Face Vid2Vid Demo',
    description = "",
    
    examples = samples
    )

iface.launch(server_name = '0.0.0.0',
    server_port = 8889)


# display
# https://github.com/tg-bomze/Face-Image-Motion-Model

import matplotlib.pyplot as plt
import base64
import numpy as np
import matplotlib.animation as animation

placeholder_bytes = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=')
placeholder_image = imageio.imread(placeholder_bytes, '.png')
placeholder_image = resize(placeholder_image, (256, 256))[..., :3]

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))
    ims = []
    for i in range(len(driving)):
        cols = [[placeholder_image], []]
        for sourceitem in source:
            cols[0].append(sourceitem)
        cols[1].append(driving[i])
        if generated is not None:
            for generateditem in generated:
                cols[1].append(generateditem[i])

        endcols = []
        for thiscol in cols:
            endcols.append(np.concatenate(thiscol, axis=1))

        im = plt.imshow(np.vstack(endcols), animated=True) # np.concatenate(cols[0], axis=1)
        plt.axis('off')
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


source_list = ['asset/source/3.png', 'asset/source/5.png', 'asset/source/4.png']
driving = 'asset/driving/1.mp4'

# saving path
os.makedirs('asset/output', exist_ok=True)

find_best_frame_ = True
free_view = False
yaw = None
pitch = None
roll = None

cpu = False
best_frame = None
relative = True
adapt_scale = True
estimate_jacobian = False


# driving
reader = imageio.get_reader(driving)
fps = reader.get_meta_data()['fps']
driving_video = []
try:
    for im in reader:
        driving_video.append(im)
except RuntimeError:
    pass
reader.close()
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

source_images = []
final = []
for idx, source in enumerate(source_list):
    # source 
    source_image = imageio.imread(source)[..., :3]
    source_image = resize(source_image, (256, 256))
    source_images.append(source_image)
    
    # inference
    if find_best_frame_ or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)

    imageio.mimsave(f'asset/output/{idx}.mp4', [img_as_ubyte(frame) for frame in predictions])
    final.append(predictions)

    
final_video = display(source_images, driving_video, final)
final_video.save(f'asset/output/final_1.mp4', fps=fps)