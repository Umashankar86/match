"""Model-independent geometry and conservative event estimates."""
from collections import Counter, defaultdict
import math

import cv2
import numpy as np

TEAM_COLORS = [(235, 185, 55), (110, 90, 240)]  # BGR


class Teams:
    def __init__(self):
        self.samples = []
        self.centers = None
        self.features = {}

    def feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 + (y2-y1)//5):min(h, y1 + (y2-y1)//2),
                     max(0, x1 + (x2-x1)//4):min(w, x2 - (x2-x1)//4)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 60)
        pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[~grass]
        return np.median(pixels, axis=0).astype(np.float32) if len(pixels) >= 5 else None

    def update(self, frame, tracks):
        for track in tracks:
            if track['role'] != 'player':
                continue
            feature = self.feature(frame, track['box'])
            if feature is not None:
                key = track['id']
                self.features[key] = .8*self.features.get(key, feature) + .2*feature
                if self.centers is None:
                    self.samples.append(feature)
        if self.centers is None and len(self.samples) >= 80:
            cv2.setRNGSeed(42)
            _, _, centers = cv2.kmeans(np.asarray(self.samples), 2, None,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .2),
                5, cv2.KMEANS_PP_CENTERS)
            if np.linalg.norm(centers[0]-centers[1]) >= 12:
                self.centers = centers
            else:
                self.samples = self.samples[-40:]
        for track in tracks:
            track['team'] = None
            if self.centers is not None and track['role'] == 'player' and track['id'] in self.features:
                track['team'] = int(np.argmin(np.linalg.norm(self.centers-self.features[track['id']], axis=1)))


def homography(image_points, confidence, world_points, threshold=.5):
    image_points = np.asarray(image_points, dtype=float)
    world = np.asarray([p if p is not None else [np.nan, np.nan] for p in world_points], dtype=float)
    if image_points.shape != world.shape:
        raise ValueError('Pitch model keypoint count does not match pitch_points.json.')
    mask = (np.asarray(confidence) >= threshold) & np.isfinite(image_points).all(1) & np.isfinite(world).all(1)
    if mask.sum() < 4:
        return None
    src, dst = image_points[mask], world[mask]
    if np.linalg.matrix_rank(dst-dst.mean(0)) < 2 or np.linalg.matrix_rank(src-src.mean(0)) < 2:
        return None
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if matrix is None or not np.isfinite(matrix).all() or inliers is None or inliers.sum() < 4 or inliers.mean() < .6:
        return None
    return matrix


def project(point, matrix, length, width):
    if matrix is None:
        return None
    v = matrix @ np.array([*point, 1.])
    if abs(v[2]) < 1e-8:
        return None
    x, y = v[:2]/v[2]
    return (float(x), float(y)) if np.isfinite([x,y]).all() and 0 <= x <= length and 0 <= y <= width else None


class Events:
    def __init__(self):
        self.candidate = None
        self.since = 0.
        self.owner = None
        self.last_seen = -math.inf
        self.possession = Counter()
        self.events = []

    def update(self, tracks, ball, t, dt, radius):
        eligible = [p for p in tracks if p['role'] in ('player', 'goalkeeper') and p['team'] is not None]
        nearest = min(eligible, key=lambda p: math.dist(p['foot'], ball), default=None) if ball is not None else None
        candidate = (nearest['id'], nearest['team']) if nearest and math.dist(nearest['foot'], ball) <= radius else None
        if candidate != self.candidate:
            self.candidate, self.since = candidate, t
        if t-self.last_seen > 1.0:
            self.owner = None
        if candidate is not None:
            self.last_seen = t
        if candidate is not None and t-self.since >= .12:
            if self.owner is not None and candidate != self.owner:
                self.events.append({'time_s': round(t,3), 'type': 'pass' if candidate[1] == self.owner[1] else 'turnover',
                                    'from_id': self.owner[0], 'to_id': candidate[0], 'team': candidate[1]})
            self.owner = candidate
            self.possession[candidate[1]] += dt
        return self.owner if candidate == self.owner else None


class Movement:
    def __init__(self):
        self.previous = {}
        self.distance = defaultdict(float)

    def update(self, key, point, t):
        old = self.previous.pop(key, None)
        if point is None:
            return None
        self.previous[key] = (point, t)
        if old:
            dt = t-old[1]
            distance = math.dist(point, old[0])
            if 0 < dt <= .5 and distance/dt <= 12:
                self.distance[key] += distance
                return distance/dt*3.6
        return None
