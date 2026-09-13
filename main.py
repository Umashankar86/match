import argparse
from pathlib import Path
from football.pipeline import Settings, run


def main():
    parser = argparse.ArgumentParser(description='Football analysis with your own trained weights')
    parser.add_argument('--video',required=True,type=Path)
    parser.add_argument('--model-dir',type=Path,default=Settings.model_dir)
    parser.add_argument('--output-dir',type=Path,default=Settings.output_dir)
    parser.add_argument('--device',default='cpu',help='cpu, 0 for NVIDIA CUDA, or mps for Apple Silicon')
    parser.add_argument('--image-size',type=int,default=1280)
    parser.add_argument('--max-frames',type=int,default=0,help='0 = whole video')
    parser.add_argument('--pitch-length',type=float,default=Settings.pitch_length)
    parser.add_argument('--pitch-width',type=float,default=Settings.pitch_width)
    parser.add_argument('--no-ball-tiles',action='store_true')
    args = parser.parse_args()
    settings = Settings(model_dir=args.model_dir,output_dir=args.output_dir,device=args.device,
        image_size=args.image_size,max_frames=args.max_frames,pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,ball_tiles=not args.no_ball_tiles)
    output = run(args.video,settings,lambda n,total,_:print(f'Processed {n}/{total or "?"} frames',end='\r'))
    print(f'\nResults: {output}')


if __name__ == '__main__':
    main()
