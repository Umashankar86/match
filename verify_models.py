"""Exercise installed weights and video export without downloading anything."""
from pathlib import Path
import json
import tempfile
import cv2
import numpy as np
from football.pipeline import Settings, run


def main():
    with tempfile.TemporaryDirectory(prefix='touchline-check-') as tmp:
        root = Path(tmp)
        source = root/'synthetic.mp4'
        writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360))
        if not writer.isOpened():
            raise RuntimeError('Cannot create the verification video.')
        for _ in range(3):
            frame = np.full((360,640,3),(40,100,40),np.uint8)
            cv2.rectangle(frame,(25,25),(615,335),(255,255,255),2)
            cv2.line(frame,(320,25),(320,335),(255,255,255),2)
            cv2.circle(frame,(320,180),50,(255,255,255),2)
            writer.write(frame)
        writer.release()
        output = run(source,Settings(output_dir=root/'outputs',image_size=640,max_frames=3))
        summary = json.loads((output/'summary.json').read_text())
        assert summary['frames']==3 and summary['pitch_enabled']
        assert len(summary['models'])==3
        capture = cv2.VideoCapture(str(output/'analysis.mp4'))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        assert decoded==3, f'Expected 3 output frames, got {decoded}'
        print(json.dumps(summary,indent=2))
        print('PASS: all three local models, tracker, tiled ball inference, pose inference, reports and video export.')
        print('Synthetic input checks integration only; football accuracy still requires match footage.')


if __name__ == '__main__':
    main()
