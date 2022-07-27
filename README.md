![supernova](https://user-images.githubusercontent.com/19509047/181189186-518d4d18-3bdb-4b9f-94fe-96cfc9b6d5ef.png)
# Supernova: Convolutional Audio Visualizer
```
usage: supernova.py [-h] --out_file F [--audio_file F] [--out_dir D] [--seed_str S] [--seed_int I] [--video_dims W H]
                    [--video_length L] [--filter_sizes S [S ...]] [--fr FR] [--sr SR] [--sensitivity S] [--interval I]
                    [--effects [E [E ...]]] [--device D] [--rand_init] [--preview] [--preserve_out_dir]

optional arguments:
  -h, --help                show this help message and exit
  --out_file F              output video file path
  --audio_file F            input audio file path
  --out_dir D               output directory for video frames [default: ./out]
  --seed_str S              seed string (hashed into 64-bit integer)
  --seed_int I              64-bit seed integer
  --video_dims W H          output video dimensions [default: (720, 720)]
  --video_length L          manually specify video length in frames
  --filter_sizes S [S ...]  convolutional filter sizes [default: (3, 5, 7)]
  --fr FR                   video frame rate [default: 30]
  --sr SR                   audio sample rate [default: 22050]
  --sensitivity S           audio reactive sensitivity [default: 1.0]
  --interval I              evolution interval for model weights (0 for no evolution) [default: 1800]
  --effects [E [E ...]]     add effects to output video
                                identity: no effect
                                invert: invert colors
                                sobel: apply Sobel filter for edge emphasis
                                glow: apply glow effect using Sobel filter overlay
                                grayscale: convert colors to grayscale
                                saturate: increase saturation
                                desaturate: decrease saturation
                                normalize: stretch contrast using top and bottom percentile
                                sblur: apply spacial blur
                                mblur: apply motion blur
                                hmirror: mirror horizontally
                                vmirror: mirror vertically
  --device D                device used for heavy computations, e.g. cpu or cuda [default: auto]
  --rand_init               initialize world from random noise
  --preview                 preview video while rendering
  --preserve_out_dir        preserve output directory after compiling video
```
