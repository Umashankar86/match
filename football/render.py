import cv2
import numpy as np
from .analytics import TEAM_COLORS


def pitch_canvas(length, width, height=300):
    scale = (height-40)/width
    canvas = np.full((height, int(length*scale)+40, 3), (40, 78, 36), np.uint8)
    def p(x,y): return (int(20+x*scale), int(20+y*scale))
    color = (175, 198, 175)
    cv2.rectangle(canvas, p(0,0), p(length,width), color, 1)
    cv2.line(canvas, p(length/2,0), p(length/2,width), color, 1)
    cv2.circle(canvas, p(length/2,width/2), int(9.15*scale), color, 1)
    for x, direction in [(0,1),(length,-1)]:
        # Match the supplied model's source template when using its 120x70 layout.
        box_depth, half_box = (20.15,20.5) if (length,width)==(120,70) else (16.5,20.16)
        cv2.rectangle(canvas, p(x,width/2-half_box), p(x+direction*box_depth,width/2+half_box), color, 1)
        cv2.rectangle(canvas, p(x,width/2-9.16), p(x+direction*5.5,width/2+9.16), color, 1)
    return canvas, p


def annotate(frame, tracks, ball, owner, events, length, width, mapping):
    view = frame.copy()
    minimap, p = pitch_canvas(length,width)
    for track in tracks:
        color = TEAM_COLORS[track['team']] if track['team'] is not None else (180,180,180)
        x1,y1,x2,y2 = map(int, track['box'])
        cv2.rectangle(view,(x1,y1),(x2,y2),color,2)
        label = f"#{track['id']} {track['role']}"
        cv2.putText(view,label,(x1,max(15,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        if owner is not None and owner[0] == track['id']:
            cv2.circle(view,(int(track['foot'][0]),int(track['foot'][1])),8,(0,255,255),2)
        if track['pitch'] is not None:
            cv2.circle(minimap,p(*track['pitch']),4,color,-1)
    if ball is not None:
        cv2.circle(view,tuple(map(int,ball)),6,(0,255,255),2)
    panel = np.full((300, max(view.shape[1],minimap.shape[1]+300),3), (23,27,32),np.uint8)
    panel[:,:minimap.shape[1]] = minimap
    total = sum(events.possession.values())
    stats = ['MATCH / ESTIMATES', 'Pitch: '+('mapped' if mapping else 'unavailable')]
    for team in range(2):
        share = f'{100*events.possession[team]/total:.1f}%' if total else '--'
        stats.append(f'Team {team+1} possession: {share}')
    stats += [f"Pass estimates: {sum(e['type']=='pass' for e in events.events)}",
              f"Turnover estimates: {sum(e['type']=='turnover' for e in events.events)}"]
    for i,line in enumerate(stats):
        cv2.putText(panel,line,(minimap.shape[1]+15,35+i*35),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,225,230),1)
    target_w = panel.shape[1]
    top = cv2.resize(view,(target_w,int(view.shape[0]*target_w/view.shape[1])))
    result = np.vstack([top,panel])
    # Most video encoders require even dimensions.
    return result[:result.shape[0]//2*2,:result.shape[1]//2*2]
