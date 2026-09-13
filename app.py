from pathlib import Path
import shutil
import uuid

import streamlit as st
from football.pipeline import ROOT, Settings, run
from football.models import model_paths

st.set_page_config(page_title='Touchline | Football analysis',page_icon='⚽',layout='wide')
st.title('⚽ Touchline')
st.caption('Your footage. Your models. Player tracking and match analysis.')
with st.sidebar:
    st.header('Analysis settings')
    model_dir = Path(st.text_input('Model folder',str(ROOT/'model')))
    device = st.selectbox('Compute device',['cpu','0','mps'],help='0 = NVIDIA GPU; mps = Apple Silicon')
    size = st.select_slider('Detection resolution',options=[640,960,1280],value=1280)
    limit = st.number_input('Maximum frames (0 = entire video)',min_value=0,value=750,step=250)
    tiled = st.checkbox('Ball detection in overlapping tiles',value=True)
    length = st.number_input('Pitch length (metres)',min_value=30.,max_value=150.,value=Settings.pitch_length)
    width = st.number_input('Pitch width (metres)',min_value=20.,max_value=100.,value=Settings.pitch_width)

cols = st.columns(3)
paths = model_paths(model_dir)
for col,name in zip(cols,['players','ball','pitch']):
    present = paths[name] is not None
    col.metric(f'{name.capitalize()} model','Found' if present else 'Awaiting your weights')
    if present:
        col.caption(paths[name].name)
mapping_present = (model_dir/'pitch_points.json').is_file()
if paths['pitch'] and not mapping_present:
    st.error('Pitch weights found, but pitch_points.json is missing from the model folder.')
else:
    st.caption('Original model filenames are supported. The included landmark mapping uses a 120 × 70 m pitch template. '
               'No models are downloaded or trained.')
upload = st.file_uploader('Upload a football video',type=['mp4','mov','avi','mkv'])
local_path = st.text_input('Or enter a local video path (for large videos)')
if st.button('Analyze video',type='primary',disabled=not (upload or local_path) or paths['players'] is None or (paths['pitch'] is not None and not mapping_present)):
    st.session_state.pop('result',None)
    temp_video = None
    try:
        if upload is not None:
            folder = ROOT/'uploads'
            folder.mkdir(exist_ok=True)
            temp_video = folder/f'{uuid.uuid4().hex}{Path(upload.name).suffix}'
            temp_video.write_bytes(upload.getbuffer())
        source = temp_video or Path(local_path)
        bar, label, preview = st.progress(0), st.empty(), st.empty()
        def update(n,total,frame):
            bar.progress(min(1.,n/total) if total else 0.)
            label.caption(f'Processed {n:,} / {total or "unknown"} frames')
            preview.image(frame,channels='BGR',width='stretch')
        with st.spinner('Analyzing video…'):
            result = run(source,Settings(model_dir=model_dir,device=device,image_size=size,
                max_frames=int(limit),ball_tiles=tiled,pitch_length=length,pitch_width=width),update)
        archive = shutil.make_archive(str(result),'zip',root_dir=result)
        st.session_state['result'] = (str(result),archive)
    except Exception as exc:
        st.error(str(exc))
    finally:
        if temp_video is not None:
            temp_video.unlink(missing_ok=True)
if 'result' in st.session_state:
    folder,archive = st.session_state['result']
    st.success(f'Analysis complete. Results saved in {folder}')
    with open(archive,'rb') as stream:
        st.download_button('Download video, reports and heatmaps',stream,file_name=Path(archive).name,mime='application/zip')
    st.caption('analysis.mp4 uses MPEG-4 video; open it in VLC if your browser cannot play it.')
    st.json((Path(folder)/'summary.json').read_text())
