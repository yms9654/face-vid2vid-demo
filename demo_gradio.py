import os
import yaml
import imageio
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
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view)
    
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
        gr.inputs.Checkbox(label="free view", default=False)        
    ],
    outputs = [
        gr.outputs.Video(label='result') # generated video
    ], 
    
    title = 'Face Vid2Vid Demo',
    description = "",
    
    examples = samples
    )

gr.close_all()
iface.launch(server_name = '0.0.0.0',
    server_port = 8889, debug=True)