from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import uuid
from datetime import datetime

import cv2
import numpy as np

from .analytics import Teams, Events, Movement, homography, project
from .render import annotate, pitch_canvas
from .models import model_paths

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    model_dir: Path = ROOT / 'model'
    output_dir: Path = ROOT / 'outputs'
    device: str = 'cpu'
    image_size: int = 1280
    confidence: float = .25
    ball_confidence: float = .15
    ball_tiles: bool = True
    pitch_length: float = 120.
    pitch_width: float = 70.
    max_frames: int = 0


def role(name):
    key = str(name).lower().replace('_','').replace(' ','').replace('-','')
    return {'player':'player','players':'player','person':'player','goalkeeper':'goalkeeper',
            'goalkeepers':'goalkeeper','gk':'goalkeeper','referee':'referee','referees':'referee',
            'ball':'ball','football':'ball','soccerball':'ball'}.get(key)


def load_models(settings):
    # Existence checks precede YOLO construction: missing weights never trigger downloads.
    folder = Path(settings.model_dir).resolve()
    paths = model_paths(folder)
    if paths['players'] is None:
        raise FileNotFoundError(f'Add your trained player model at {folder / "players.pt"}.')
    mapping = None
    if paths['pitch'] is not None:
        mapping_file = folder/'pitch_points.json'
        if not mapping_file.is_file():
            raise ValueError('pitch.pt needs model/pitch_points.json. See model/README.md for the landmark format.')
        mapping = json.loads(mapping_file.read_text(encoding='utf-8'))
        if not isinstance(mapping,list) or sum(p is not None for p in mapping) < 4:
            raise ValueError('Pitch mapping must be a list with at least four coordinate pairs.')
        for p in mapping:
            if p is not None and (not isinstance(p,list) or len(p)!=2 or
                not all(isinstance(v,(int,float)) and math.isfinite(v) for v in p) or
                not 0 <= p[0] <= settings.pitch_length or not 0 <= p[1] <= settings.pitch_width):
                raise ValueError('Pitch coordinates must be finite [x,y] pairs within configured pitch dimensions.')
    from ultralytics import YOLO
    models = {name: YOLO(str(path)) if path is not None else None for name,path in paths.items()}
    for name in ('players','ball'):
        model = models[name]
        if model is not None:
            if model.task != 'detect':
                raise ValueError(f'{name}.pt must be an object detection model.')
            recognized = {role(n) for n in model.names.values()}
            if ('player' if name=='players' else 'ball') not in recognized:
                raise ValueError(f'{name}.pt has unsupported class names: {model.names}. Update role() aliases if needed.')
    if models['pitch'] is not None and models['pitch'].task != 'pose':
        raise ValueError('pitch.pt must be a pose/keypoint model.')
    if models['pitch'] is not None:
        shape = getattr(models['pitch'].model, 'kpt_shape', None)
        if shape is None or shape[0] != len(mapping) or shape[1] != 3:
            raise ValueError('Pitch model must have one [x,y,visibility] keypoint per entry in pitch_points.json.')
    return models, mapping


def ball_detection(model, frame, settings):
    if model is None:
        return None
    h,w = frame.shape[:2]
    candidates = []
    def starts(size):
        return sorted(set(list(range(0,max(1,size-640+1),512))+[max(0,size-640)]))
    tiles = [(x,y,frame[y:y+640,x:x+640]) for y in starts(h) for x in starts(w)] if settings.ball_tiles else [(0,0,frame)]
    for x,y,tile in tiles:
        result = model.predict(tile,imgsz=640,conf=settings.ball_confidence,device=settings.device,verbose=False)[0]
        for box in result.boxes:
            if role(result.names[int(box.cls.item())]) == 'ball':
                x1,y1,x2,y2 = box.xyxy[0].cpu().tolist()
                candidates.append((float(box.conf.item()),(x+(x1+x2)/2,y+(y1+y2)/2)))
    return max(candidates,key=lambda v:v[0])[1] if candidates else None


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(video, settings=None, progress=None):
    settings = settings or Settings()
    video = Path(video).resolve()
    if not video.is_file():
        raise FileNotFoundError(f'Video does not exist: {video}')
    if settings.pitch_length <= 0 or settings.pitch_width <= 0 or settings.max_frames < 0:
        raise ValueError('Pitch dimensions must be positive and max_frames nonnegative.')
    if settings.image_size < 32 or not 0 < settings.confidence <= 1 or not 0 < settings.ball_confidence <= 1:
        raise ValueError('Invalid image size or confidence threshold.')
    models, world = load_models(settings)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError('Cannot decode this video. Try an MP4 with H.264 video.')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Video has no valid frame rate; convert it to a constant-frame-rate MP4.')
    count = max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    total = min(count,settings.max_frames) if settings.max_frames and count else count
    output = Path(settings.output_dir)/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True,exist_ok=False)
    teams, events, movement = Teams(), Events(), Movement()
    histories = {}
    trajectory_canvases = {}
    trajectory_previous = {}
    position_sums = {}
    track_meta = {}
    writer = None
    index = 0
    fields = ['frame','time_s','track_id','role','team','confidence','x1','y1','x2','y2','pitch_x_m','pitch_y_m','speed_kmh']
    try:
        with (output/'tracks.csv').open('w',newline='',encoding='utf-8') as stream:
            csv_writer = csv.DictWriter(stream,fieldnames=fields)
            csv_writer.writeheader()
            while not settings.max_frames or index < settings.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                t = index/fps
                result = models['players'].track(frame,persist=True,tracker=str(ROOT/'configs/botsort.yaml'),
                    device=settings.device,imgsz=settings.image_size,conf=settings.confidence,verbose=False)[0]
                tracks, fallback_ball = [], None
                best_ball_conf = -1
                for box in result.boxes:
                    kind = role(result.names[int(box.cls.item())])
                    coords = box.xyxy[0].cpu().tolist()
                    confidence = float(box.conf.item())
                    if kind == 'ball' and confidence > best_ball_conf:
                        fallback_ball = ((coords[0]+coords[2])/2,(coords[1]+coords[3])/2)
                        best_ball_conf = confidence
                    if kind not in ('player','goalkeeper','referee') or box.id is None:
                        continue
                    tracks.append({'id':int(box.id.item()),'role':kind,'box':coords,'confidence':confidence,
                                   'foot':((coords[0]+coords[2])/2,coords[3])})
                ball = ball_detection(models['ball'],frame,settings) if models['ball'] is not None else fallback_ball
                teams.update(frame,tracks)
                matrix = None
                if models['pitch'] is not None:
                    pose = models['pitch'].predict(frame,device=settings.device,imgsz=640,conf=.25,verbose=False)[0]
                    if pose.keypoints is not None and len(pose.keypoints.xy):
                        best = int(pose.boxes.conf.argmax().item())
                        if pose.keypoints.conf is None:
                            raise ValueError('Pitch model must return keypoint confidence/visibility values.')
                        matrix = homography(pose.keypoints.xy[best].cpu().numpy(),pose.keypoints.conf[best].cpu().numpy(),world)
                for track in tracks:
                    point = project(track['foot'],matrix,settings.pitch_length,settings.pitch_width)
                    track['pitch'] = point
                    speed = movement.update(track['id'],point,t)
                    track_meta[track['id']] = {'role':track['role'],'team':track['team']}
                    if point is not None:
                        # Histograms bound memory even for full-match videos.
                        histogram = histories.setdefault(track['id'],np.zeros((68,105),np.float32))
                        histogram[min(67,int(point[1]/settings.pitch_width*68)),min(104,int(point[0]/settings.pitch_length*105))] += 1
                        sums = position_sums.setdefault(track['id'],[0.,0.,0])
                        sums[0] += point[0]
                        sums[1] += point[1]
                        sums[2] += 1
                        if track['id'] not in trajectory_canvases:
                            trajectory_canvases[track['id']] = pitch_canvas(settings.pitch_length,settings.pitch_width)[0]
                        canvas = trajectory_canvases[track['id']]
                        scale = (canvas.shape[0]-40)/settings.pitch_width
                        pixel = tuple(int(20+v*scale) for v in point)
                        previous = trajectory_previous.get(track['id'])
                        if previous is not None and speed is not None:
                            cv2.line(canvas,previous,pixel,(40,215,250),1)
                        trajectory_previous[track['id']] = pixel
                    else:
                        trajectory_previous.pop(track['id'],None)
                    csv_writer.writerow(dict(zip(fields,[index,round(t,3),track['id'],track['role'],track['team'],
                        round(track['confidence'],4),*track['box'],*(point or ('','')),speed if speed is not None else ''])))
                owner = events.update(tracks,ball,t,1/fps,max(15,frame.shape[1]*.035))
                rendered = annotate(frame,tracks,ball,owner,events,settings.pitch_length,settings.pitch_width,matrix is not None)
                if writer is None:
                    rh,rw = rendered.shape[:2]
                    writer = cv2.VideoWriter(str(output/'analysis.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(rw,rh))
                    if not writer.isOpened():
                        raise RuntimeError('No working MP4 encoder in OpenCV.')
                writer.write(rendered)
                index += 1
                if progress and (index == 1 or index%10 == 0):
                    progress(index,total,rendered)
        if not index:
            raise ValueError('Video contains no decodable frames.')
    except Exception as exc:
        (output/'FAILED.txt').write_text(f'Incomplete run after {index} frames: {exc}',encoding='utf-8')
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    write_csv(output/'events.csv',events.events,['time_s','type','from_id','to_id','team'])
    write_csv(output/'players.csv',[{'track_id':key,**meta,'observed_distance_m':round(movement.distance[key],2)}
        for key,meta in track_meta.items()],['track_id','role','team','observed_distance_m'])
    for key,hist in histories.items():
        canvas, _ = pitch_canvas(settings.pitch_length,settings.pitch_width)
        density = cv2.GaussianBlur(hist,(9,9),0)
        heat = cv2.applyColorMap((density/max(float(density.max()),1e-6)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        heat = cv2.resize(heat,(canvas.shape[1]-40,canvas.shape[0]-40))
        canvas[20:-20,20:-20] = cv2.addWeighted(canvas[20:-20,20:-20],.45,heat,.55,0)
        cv2.imwrite(str(output/f'heatmap_{key}.png'),canvas)
        cv2.imwrite(str(output/f'trajectory_{key}.png'),trajectory_canvases[key])
    for team in range(2):
        network, p = pitch_canvas(settings.pitch_length,settings.pitch_width)
        from collections import Counter
        edges = Counter((e['from_id'],e['to_id']) for e in events.events if e['type']=='pass' and e['team']==team)
        for (source,target),number in edges.items():
            if source in position_sums and target in position_sums:
                a,b = position_sums[source],position_sums[target]
                cv2.arrowedLine(network,p(a[0]/a[2],a[1]/a[2]),p(b[0]/b[2],b[1]/b[2]),(50,190,245),min(6,number),tipLength=.15)
        for key,(x,y,n) in position_sums.items():
            if track_meta[key]['team'] == team:
                pixel = p(x/n,y/n)
                cv2.circle(network,pixel,5,(235,235,235),-1)
                cv2.putText(network,str(key),(pixel[0]+6,pixel[1]),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
        if position_sums:
            cv2.imwrite(str(output/f'pass_network_team_{team}.png'),network)
    summary = {'frames':index,'fps':fps,'duration_s':index/fps,'tracks':len(track_meta),
        'possession_seconds':dict(events.possession),'estimated_passes':sum(e['type']=='pass' for e in events.events),
        'estimated_turnovers':sum(e['type']=='turnover' for e in events.events),
        'pitch_enabled':models['pitch'] is not None,
        'pitch_dimensions_m':[settings.pitch_length,settings.pitch_width],
        'models':{k:str(v) for k,v in model_paths(settings.model_dir).items() if v is not None},
        'notes':['No audio in analysis video.','Team IDs are jersey clusters, not club names.',
                 'Goalkeepers remain unassigned because their kits differ.',
                 'Possession and events are proximity estimates; unknown periods are excluded.',
                 'Distance is observed valid movement only, not full-match running distance.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if progress:
        progress(index,index,rendered)
    return output
