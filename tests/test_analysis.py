import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from football.analytics import Events, Movement, homography, project
from football.pipeline import Settings, load_models, run
from football.models import MODEL_NAMES, model_paths


class AnalyticsTests(unittest.TestCase):
    def test_original_filenames_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for names in MODEL_NAMES.values():
                (root/names[1]).touch()
            paths = model_paths(root)
            self.assertTrue(all(p is not None for p in paths.values()))
            self.assertEqual(paths['pitch'].name,'pitch_kpts32_y8s_640_e500_AO.pt')
            (root/'players.pt').touch()
            self.assertEqual(model_paths(root)['players'].name,'players.pt')

    def test_included_landmarks_and_defaults(self):
        root = Path(__file__).resolve().parents[1]
        points = json.loads((root/'model/pitch_points.json').read_text())
        self.assertEqual(len(points),32)
        self.assertEqual(points[0],[0,0])
        self.assertEqual(points[29],[Settings.pitch_length,Settings.pitch_width])
        self.assertEqual(points[30:],[[50.85,35],[69.15,35]])
        self.assertEqual(points[9],[20.15,14.5])

    def test_projection_and_degenerate_landmarks(self):
        image = np.array([[0,0],[200,0],[200,100],[0,100]],np.float32)
        world = [[0,0],[100,0],[100,50],[0,50]]
        matrix = homography(image,np.ones(4),world)
        np.testing.assert_allclose(project((100,50),matrix,100,50),(50,25))
        self.assertIsNone(project((300,50),matrix,100,50))
        self.assertIsNone(homography(image,np.zeros(4),world))
        self.assertIsNone(homography(image,np.ones(4),[[0,0],[1,0],[2,0],[3,0]]))
        with self.assertRaises(ValueError):
            homography(image,np.ones(4),world[:3])

    def test_movement_rejects_jumps_and_missing_geometry(self):
        movement = Movement()
        self.assertIsNone(movement.update(1,(0,0),0))
        self.assertAlmostEqual(movement.update(1,(1,0),.2),18)
        self.assertIsNone(movement.update(1,(90,0),.3))
        self.assertEqual(movement.distance[1],1)
        movement.update(1,None,.4)
        self.assertIsNone(movement.update(1,(1,0),.5))

    def test_possession_pass_and_unknown_gap(self):
        events = Events()
        players = [{'id':1,'role':'player','team':0,'foot':(0,0)},
                   {'id':2,'role':'player','team':0,'foot':(100,0)},
                   {'id':3,'role':'player','team':1,'foot':(200,0)}]
        for t in [0,.1,.2]: events.update(players,(0,0),t,.1,20)
        for t in [.3,.4,.5]: events.update(players,(100,0),t,.1,20)
        self.assertEqual([e['type'] for e in events.events],['pass'])
        events.update(players,None,2,.1,20)
        for t in [2.1,2.2,2.3]: events.update(players,(200,0),t,.1,20)
        self.assertEqual(len(events.events),1)

    def test_missing_models_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError,'players.pt'):
                load_models(Settings(model_dir=Path(tmp)))

    def test_video_pipeline_without_real_weights(self):
        class Result:
            boxes = []
        class FakePlayer:
            def track(self,*args,**kwargs): return [Result()]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input.mp4'
            writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,180))
            self.assertTrue(writer.isOpened())
            for _ in range(5): writer.write(np.full((180,320,3),70,np.uint8))
            writer.release()
            with patch('football.pipeline.load_models',return_value=({'players':FakePlayer(),'ball':None,'pitch':None},None)):
                output = run(source,Settings(output_dir=root/'outputs'))
            summary = json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['frames'],5)
            self.assertEqual(summary['estimated_passes'],0)
            cap = cv2.VideoCapture(str(output/'analysis.mp4'))
            frames = 0
            while cap.read()[0]: frames += 1
            cap.release()
            self.assertEqual(frames,5)
            for name in ['tracks.csv','events.csv','players.csv']:
                self.assertTrue((output/name).read_text().strip())

    def test_interface_starts_with_no_models(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(len(app.metric),3)

    def test_mapped_tracks_generate_reports(self):
        # Minimal tensor adapters emulate only the public result properties used.
        class Tensor:
            def __init__(self,data): self.data = np.asarray(data)
            def __getitem__(self,key): return Tensor(self.data[key])
            def cpu(self): return self
            def numpy(self): return self.data
            def tolist(self): return self.data.tolist()
            def item(self): return self.data.item()
            def argmax(self): return Tensor(self.data.argmax())
            def __len__(self): return len(self.data)
        class Box:
            cls = Tensor(0)
            conf = Tensor(.9)
            id = Tensor(1)
            xyxy = Tensor([[80,50,100,100]])
        class Result:
            boxes = [Box()]
            names = {0:'player'}
        class FakePlayer:
            def track(self,*args,**kwargs): return [Result()]
        class Keypoints:
            xy = Tensor([[[0,0],[320,0],[320,180],[0,180]]])
            conf = Tensor([[1,1,1,1]])
        class PoseBoxes:
            conf = Tensor([.9])
        class Pose:
            boxes = PoseBoxes()
            keypoints = Keypoints()
        class FakePitch:
            def predict(self,*args,**kwargs): return [Pose()]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input.mp4'
            writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,180))
            self.assertTrue(writer.isOpened())
            for _ in range(5): writer.write(np.full((180,320,3),70,np.uint8))
            writer.release()
            fake = ({'players':FakePlayer(),'ball':None,'pitch':FakePitch()},[[0,0],[105,0],[105,68],[0,68]])
            with patch('football.pipeline.load_models',return_value=fake):
                output = run(source,Settings(output_dir=root/'outputs'))
            import csv
            with (output/'tracks.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),5)
            self.assertAlmostEqual(float(rows[0]['pitch_x_m']),90/320*105,places=3)
            self.assertTrue((output/'heatmap_1.png').is_file())
            self.assertTrue((output/'trajectory_1.png').is_file())
            self.assertTrue((output/'pass_network_team_0.png').is_file())


if __name__ == '__main__':
    unittest.main()
