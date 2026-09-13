"""Resolve only local weights, retaining the user's original filenames."""
from pathlib import Path

MODEL_NAMES = {
    'players': ('players.pt', 'best_players_gk_1280_s_e300.pt'),
    'ball': ('ball.pt', 'ball_tracking_1280_e300.pt'),
    'pitch': ('pitch.pt', 'pitch_kpts32_y8s_640_e500_AO.pt'),
}


def model_paths(folder):
    folder = Path(folder).expanduser().resolve()
    return {role: next((folder/name for name in names if (folder/name).is_file()), None)
            for role, names in MODEL_NAMES.items()}
