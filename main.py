from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from glob import glob

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np


@dataclass(frozen=True)
class Cfg:
    W: int = 448
    H: int = 288
    crop: int = 64
    f_px: float = 230.0
    baseline_m: float = 0.12
    cam_height_m: float = 1.6
    num_disp: int = 64
    ds: int = 4
    danger_m: float = 1.0
    caution_m: float = 2.0
    min_zone_px: int = 500
    tilt_px: float = 0.0

    @property
    def W_eff(self) -> int:
        return self.W - self.crop

    @property
    def cx(self) -> float:
        return self.crop + self.W_eff / 2

    @property
    def cy(self) -> float:
        return self.H / 2


CFG = Cfg()
CLASS_NAMES = ["clear", "caution", "danger"]
ZONE_NAMES = ["left", "ahead", "right"]


def zone_slices(width: int, n: int = 3):
    edges = np.linspace(0, width, n + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n)]


@dataclass
class Obstacle:
    Z: float
    X: float
    width: float
    height: float
    bottom: float

    def box(self, cfg: Cfg = CFG, dy: float = 0.0):

        f, h = cfg.f_px, cfg.cam_height_m
        x0 = cfg.cx + f * (self.X - self.width / 2) / self.Z
        x1 = cfg.cx + f * (self.X + self.width / 2) / self.Z
        y1 = cfg.cy + dy + f * (h - self.bottom) / self.Z
        y0 = cfg.cy + dy + f * (h - self.bottom - self.height) / self.Z
        return (int(np.clip(round(x0), 0, cfg.W)), int(np.clip(round(x1), 0, cfg.W)),
                int(np.clip(round(y0), 0, cfg.H)), int(np.clip(round(y1), 0, cfg.H)))


def sample_obstacles(rng: np.random.Generator, cfg: Cfg = CFG) -> list[Obstacle]:
    out = []
    for _ in range(rng.choice(5, p=[0.2, 0.3, 0.25, 0.15, 0.1])):
        Z = rng.uniform(0.6, 2.4) if rng.random() < 0.55 else rng.uniform(2.4, 7.0)
        X = float(np.clip(rng.uniform(-0.8, 0.8) * Z, -2.5, 2.5))
        kind = rng.choice(["person", "box", "wall", "overhead"], p=[0.4, 0.3, 0.1, 0.2])
        if kind == "person":
            w, h, b = rng.uniform(0.3, 0.7), rng.uniform(1.4, 1.9), 0.0
        elif kind == "box":
            w, h, b = rng.uniform(0.3, 1.2), rng.uniform(0.3, 0.9), 0.0
        elif kind == "wall":
            w, h, b = rng.uniform(1.5, 3.5), 2.2, 0.0
        else:
            w, h, b = rng.uniform(0.4, 1.5), rng.uniform(0.3, 0.8), rng.uniform(1.0, 1.4)
        out.append(Obstacle(Z, X, w, h, b))
    return out


def _texture(rng, h, w, strength=1.0, base=128.0):

    t = np.zeros((h, w), np.float32)
    for sigma, amp in ((0.8, 1.0), (2.5, 1.0), (7.0, 0.8)):
        n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        t += amp * n / (n.std() + 1e-6)
    t /= t.std() + 1e-6
    return np.clip(base + strength * 45.0 * t, 0, 255)


def render_stereo_pair(rng, obstacles, cfg: Cfg = CFG, dy: float = 0.0):

    H, W = cfg.H, cfg.W

    d_wall = cfg.f_px * cfg.baseline_m / rng.uniform(5, 9)
    rows = np.arange(H, dtype=np.float32)[:, None]
    d_floor = np.clip(cfg.baseline_m * (rows - cfg.cy - dy) / cfg.cam_height_m, 0, None)
    disp = np.maximum(d_wall, d_floor) * np.ones((1, W), np.float32)

    left = _texture(rng, H, W, rng.uniform(0.6, 1.0), rng.uniform(90, 160))
    mask = np.zeros((H, W), bool)
    for ob in sorted(obstacles, key=lambda o: -o.Z):
        x0, x1, y0, y1 = ob.box(cfg, dy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        left[y0:y1, x0:x1] = _texture(rng, y1 - y0, x1 - x0, rng.uniform(0.3, 1.0), rng.uniform(40, 220))
        disp[y0:y1, x0:x1] = cfg.f_px * cfg.baseline_m / ob.Z
        mask[y0:y1, x0:x1] = True
    disp = np.rint(disp)


    right = _texture(rng, H, W, 0.8, 128.0)
    rr = np.broadcast_to(np.arange(H)[:, None], (H, W))
    cc = np.rint(np.arange(W)[None, :] - disp).astype(np.int32)
    ok = (cc >= 0) & (cc < W)
    r, c, d, v = rr[ok], cc[ok], disp[ok], left[ok]
    zbuf = np.full((H, W), -1.0, np.float32)
    np.maximum.at(zbuf, (r, c), d)
    win = d >= zbuf[r, c]
    right[r[win], c[win]] = v[win]


    right = right * rng.uniform(0.92, 1.08) + rng.uniform(-6, 6)
    sigma = rng.uniform(1, 4)
    left = left + rng.normal(0, sigma, left.shape)
    right = right + rng.normal(0, sigma, right.shape)
    to8 = lambda a: np.clip(a, 0, 255).astype(np.uint8)
    return to8(left), to8(right), disp.astype(np.float32), mask


def zone_labels(disp_gt, obst_mask, cfg: Cfg = CFG):

    depth = cfg.f_px * cfg.baseline_m / np.maximum(disp_gt, 1e-3)
    labels = np.zeros(3, np.int64)
    nearest = np.full(3, np.inf, np.float32)
    for z, (a, b) in enumerate(zone_slices(disp_gt.shape[1])):
        m = obst_mask[:, a:b]
        if m.sum() < cfg.min_zone_px:
            continue
        nearest[z] = np.percentile(depth[:, a:b][m], 2)
        labels[z] = 2 if nearest[z] < cfg.danger_m else 1 if nearest[z] < cfg.caution_m else 0
    return labels, nearest


class StereoRectifier:


    def __init__(self, calib_path: str):
        c = np.load(calib_path)
        size = (int(c["w"]), int(c["h"]))
        self.map_l = cv2.initUndistortRectifyMap(c["K1"], c["D1"], c["R1"], c["P1"], size, cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(c["K2"], c["D2"], c["R2"], c["P2"], size, cv2.CV_32FC1)
        self.f_px = float(c["P2"][0, 0])
        self.baseline_m = float(-c["P2"][0, 3] / c["P2"][0, 0])

    def apply(self, left, right):
        return (cv2.remap(left, *self.map_l, cv2.INTER_LINEAR),
                cv2.remap(right, *self.map_r, cv2.INTER_LINEAR))


class StereoDepth:


    def __init__(self, cfg: Cfg = CFG):
        self.cfg = cfg
        bs = 5
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=cfg.num_disp, blockSize=bs,
            P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=120, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, left, right):

        d = self.sgbm.compute(left, right).astype(np.float32) / 16.0
        d = d[:, self.cfg.crop:]
        valid = d >= 1.0
        return np.where(valid, d, 0.0).astype(np.float32), valid


def depth_from_disparity(disp, cfg: Cfg = CFG):
    return cfg.f_px * cfg.baseline_m / np.maximum(disp, 1e-3)


def make_features(disp, valid, cfg: Cfg = CFG):


    ds = cfg.ds
    H, W = disp.shape
    blocks = (disp / 32.0).reshape(H // ds, ds, W // ds, ds)
    v = valid.astype(np.float32).reshape(H // ds, ds, W // ds, ds)
    vsum = v.sum(axis=(1, 3))
    dmean = blocks.sum(axis=(1, 3)) / np.maximum(vsum, 1)
    dmax = blocks.max(axis=(1, 3))
    vfrac = vsum / (ds * ds)
    rowc = np.broadcast_to(np.linspace(0, 1, H // ds, dtype=np.float32)[:, None], dmean.shape)
    return np.stack([dmean, dmax, vfrac, rowc], -1).astype(np.float32)


def baseline_zone_stats(disp, valid, cfg: Cfg = CFG):


    H, W = disp.shape
    rows = np.arange(H, dtype=np.float32)[:, None]
    d_floor = np.clip(cfg.baseline_m * (rows - cfg.cy) / cfg.cam_height_m, 0, None)
    solid = valid & (disp > d_floor + 2.0)
    depth = depth_from_disparity(disp, cfg)
    levels = np.zeros(3, np.int64)
    nearest = np.full(3, np.inf, np.float32)
    for z, (a, b) in enumerate(zone_slices(W)):
        m = solid[:, a:b]
        if m.sum() < cfg.min_zone_px:
            continue
        nearest[z] = np.percentile(depth[:, a:b][m], 5)
        levels[z] = 2 if nearest[z] < cfg.danger_m else 1 if nearest[z] < cfg.caution_m else 0
    return levels, nearest


def make_sample(seed, cfg: Cfg = CFG, stereo: StereoDepth | None = None):
    stereo = stereo or StereoDepth(cfg)
    rng = np.random.default_rng(seed)
    obstacles = sample_obstacles(rng, cfg)
    dy = rng.uniform(-cfg.tilt_px, cfg.tilt_px)
    left, right, d_gt, mask = render_stereo_pair(rng, obstacles, cfg, dy)
    disp, valid = stereo.disparity(left, right)
    y, near_gt = zone_labels(d_gt[:, cfg.crop:], mask[:, cfg.crop:], cfg)
    y_base, near_base = baseline_zone_stats(disp, valid, cfg)
    return make_features(disp, valid, cfg), y, y_base, near_gt, near_base


def build_dataset(n, seed0, cache, cfg: Cfg = CFG):

    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    stereo = StereoDepth(cfg)
    cols = [[] for _ in range(5)]
    t0 = time.time()
    for i in range(n):
        for col, item in zip(cols, make_sample(seed0 + i, cfg, stereo)):
            col.append(item)
        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{n} samples ({time.time() - t0:.0f}s)")
    X, y, yb, ng, nb = (np.stack(c) for c in cols)
    data = dict(X=X.astype(np.float16), y=y, y_base=yb, near_gt=ng, near_base=nb)
    np.savez_compressed(cache, **data)
    return data


def build_model(cfg: Cfg = CFG):


    import tensorflow as tf
    L = tf.keras.layers
    h, w = cfg.H // cfg.ds, cfg.W_eff // cfg.ds
    inp = L.Input((h, w, 4), name="disparity_features")
    x = inp
    for filters in (16, 32, 64):
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D(2)(x)
    x = L.Conv2D(64, (h // 8, 1), activation="relu")(x)
    x = L.Reshape((3, -1))(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.2)(x)
    out = L.Dense(3, activation="softmax", name="zone_risk")(x)
    return tf.keras.Model(inp, out, name="koru_zone_risk")


def make_loss(class_weights):

    import tensorflow as tf
    w = tf.constant(class_weights, tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.int32), tf.shape(y_pred)[:2])
        p_true = tf.gather(y_pred, y_true, batch_dims=2)
        ce = -tf.math.log(tf.clip_by_value(p_true, 1e-7, 1.0))
        return tf.reduce_mean(ce * tf.gather(w, y_true), axis=-1)
    return loss


def report(name, y_true, y_pred):
    danger, clear = y_true == 2, y_true == 0
    recall = (y_pred[danger] == 2).mean() if danger.any() else float("nan")
    missed = (y_pred[danger] == 0).mean() if danger.any() else float("nan")
    false_alarm = (y_pred[clear] == 2).mean() if clear.any() else float("nan")
    print(f"  {name:<10} accuracy={(y_true == y_pred).mean():.3f}   danger recall={recall:.3f}   "
          f"danger->clear (worst)={missed:.3f}   clear->danger (false alarm)={false_alarm:.3f}")


def train(n_train, n_val, epochs, out_path, export=True):
    import tensorflow as tf
    tr = build_dataset(n_train, 0, f"koru_train_{n_train}.npz")
    va = build_dataset(n_val, 10_000_000, f"koru_val_{n_val}.npz")
    Xtr, ytr = tr["X"].astype(np.float32), tr["y"]
    Xva, yva = va["X"].astype(np.float32), va["y"]

    Xtr = np.concatenate([Xtr, Xtr[:, :, ::-1]])
    ytr = np.concatenate([ytr, ytr[:, ::-1]])
    print("train class counts (clear/caution/danger):", np.bincount(ytr.ravel(), minlength=3))

    counts = np.maximum(np.bincount(ytr.ravel(), minlength=3), 1)
    cw = np.sqrt(counts.sum() / (3 * counts))
    cw = cw / cw.mean()

    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=make_loss(cw))
    model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=32, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=4, restore_best_weights=True)])
    model.save(out_path)
    print(f"saved {out_path}")

    pred = model.predict(Xva, verbose=0).argmax(-1)
    print("\nValidation results (per zone):")
    report("baseline", yva, va["y_base"])
    report("koru-cnn", yva, pred)
    if export:
        export_tflite(out_path, out_path.replace(".keras", ".tflite"), Xva)


def export_tflite(keras_path, tflite_path, sample_x):

    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(tflite_path, "wb") as f:
        f.write(conv.convert())
    k, t = KoruModel(keras_path), KoruModel(tflite_path)
    diff = max(float(np.abs(k.predict_probs(x) - t.predict_probs(x)).max()) for x in sample_x[:20])
    print(f"exported {tflite_path} ({os.path.getsize(tflite_path) / 1024:.0f} KB), "
          f"max difference vs Keras on 20 samples: {diff:.4f}")


class KoruModel:


    def __init__(self, path):
        import tensorflow as tf
        self.keras = None
        if path.endswith(".tflite"):
            self.interp = tf.lite.Interpreter(model_path=path)
            self.interp.allocate_tensors()
            self.i_in = self.interp.get_input_details()[0]["index"]
            self.i_out = self.interp.get_output_details()[0]["index"]
        else:
            self.keras = tf.keras.models.load_model(path, compile=False)

    def predict_probs(self, feat):
        x = feat[None].astype(np.float32)
        if self.keras is not None:
            return self.keras(x, training=False).numpy()[0]
        self.interp.set_tensor(self.i_in, x)
        self.interp.invoke()
        return self.interp.get_tensor(self.i_out)[0]


@dataclass
class Alert:
    text: str
    level: int
    zone: int
    haptic: tuple


class AlertEngine:


    def __init__(self, alpha=0.5, cooldown_s=None):
        self.alpha = alpha
        self.cooldown = cooldown_s or {1: 2.5, 2: 1.0}
        self.p = np.tile([1.0, 0.0, 0.0], (3, 1))
        self.levels = [0, 0, 0]
        self.last = {}
        self.last_hazard_t = -np.inf

    @staticmethod
    def level_from_probs(p):
        if p[2] >= 0.40:
            return 2
        return 1 if p[1] + p[2] >= 0.50 else 0

    def update(self, probs, nearest_m, now):
        self.p = self.alpha * probs + (1 - self.alpha) * self.p
        self.levels = [self.level_from_probs(p) for p in self.p]
        if max(self.levels) == 0:
            if now - self.last_hazard_t > 1.0:
                self.last.clear()
            return None
        self.last_hazard_t = now
        zone = max(range(3), key=lambda z: (self.levels[z], z == 1))
        level = self.levels[zone]
        prev = self.last.get(zone)
        if prev and prev[0] >= level and now - prev[1] < self.cooldown[level]:
            return None
        self.last[zone] = (level, now)
        where = ["on your left", "ahead", "on your right"][zone]
        dist = f", {nearest_m[zone]:.1f} meters" if np.isfinite(nearest_m[zone]) else ""
        text = ("Stop! Obstacle " if level == 2 else "Obstacle ") + where + dist
        return Alert(text, level, zone, tuple(l / 2 for l in self.levels))


class Speaker:


    def __init__(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
        except Exception:
            self.engine = None

    def say(self, text):
        print(f"[KORU] {text}")
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()


class KoruPipeline:
    def __init__(self, cfg: Cfg = CFG, model: KoruModel | None = None):
        self.cfg, self.model = cfg, model
        self.stereo = StereoDepth(cfg)
        self.alerts = AlertEngine()

    def process(self, left_gray, right_gray, now):
        disp, valid = self.stereo.disparity(left_gray, right_gray)
        levels, nearest = baseline_zone_stats(disp, valid, self.cfg)
        if self.model is not None:
            probs = self.model.predict_probs(make_features(disp, valid, self.cfg))
        else:
            probs = np.eye(3)[levels]
        alert = self.alerts.update(probs, nearest, now)
        return alert, dict(disp=disp, nearest=nearest, probs=probs, levels=self.alerts.levels)


def visualize(left, disp, levels, cfg: Cfg = CFG):

    gray = cv2.cvtColor(left[:, cfg.crop:], cv2.COLOR_GRAY2BGR)
    dmap = cv2.applyColorMap((np.clip(disp / 48.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dmap[disp == 0] = 0
    colours = [(80, 200, 80), (0, 165, 255), (60, 60, 230)]
    for img in (gray, dmap):
        for z, (a, b) in enumerate(zone_slices(img.shape[1])):
            cv2.rectangle(img, (a, 0), (b - 1, 26), colours[levels[z]], -1)
            cv2.putText(img, f"{ZONE_NAMES[z]}: {CLASS_NAMES[levels[z]]}", (a + 6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if a:
                cv2.line(img, (a, 27), (a, img.shape[0]), (255, 255, 255), 1)
    return np.hstack([gray, dmap])


def run_demo(model_path, out_png="koru_demo.png"):
    cfg = CFG
    model = KoruModel(model_path) if model_path and os.path.exists(model_path) else None
    print("model:", model_path if model else "none found -> using the rule-based detector only")
    pipe, speaker = KoruPipeline(cfg, model), Speaker()

    scenario = [(5.0, 0.15, 0.5, 1.7, 0.0), (3.6, 0.9, 0.6, 0.5, 0.0), (4.2, -0.9, 1.0, 0.5, 1.2)]
    step_m, dt = 0.24, 0.2
    snapshots, t = [], 0
    print(f"\n{'time':>5} {'person Z':>9}   estimated nearest L/A/R [m]      zone levels      alert")
    while scenario[0][0] - t * step_m > 0.7:
        obstacles = [Obstacle(Z0 - t * step_m, X, w, h, b) for Z0, X, w, h, b in scenario if Z0 - t * step_m > 0.5]
        left, right, _, _ = render_stereo_pair(np.random.default_rng(500 + t), obstacles, cfg)
        alert, info = pipe.process(left, right, now=t * dt)
        near = "/".join(f"{v:4.1f}" if v < 6 else " -- " for v in info["nearest"])
        lv = "/".join(CLASS_NAMES[l][:4] for l in info["levels"])
        print(f"{t * dt:5.1f} {scenario[0][0] - t * step_m:9.2f}   {near:<30} {lv:<16} {alert.text if alert else ''}")
        if alert:
            if speaker.engine:
                speaker.say(alert.text)
            snapshots.append(visualize(left, info["disp"], info["levels"], cfg))
        t += 1
    if snapshots:
        picks = [snapshots[i] for i in sorted({0, len(snapshots) // 2, len(snapshots) - 1})]
        cv2.imwrite(out_png, np.vstack(picks))
        print(f"\nsaved {out_png} (camera view | disparity map, at moments when an alert was spoken)")


def run_live(a):
    cfg, rect = CFG, None
    if a.calib:
        rect = StereoRectifier(a.calib)
        cfg = replace(CFG, f_px=rect.f_px, baseline_m=rect.baseline_m)
    model = KoruModel(a.model) if a.model and os.path.exists(a.model) else None
    pipe, speaker = KoruPipeline(cfg, model), Speaker()
    cap_l, cap_r = cv2.VideoCapture(a.left), cv2.VideoCapture(a.right)
    while cap_l.grab() and cap_r.grab():
        _, fl = cap_l.retrieve()
        _, fr = cap_r.retrieve()
        gl, gr = (cv2.cvtColor(cv2.resize(f, (cfg.W, cfg.H)), cv2.COLOR_BGR2GRAY) for f in (fl, fr))
        if rect:
            gl, gr = rect.apply(gl, gr)
        alert, info = pipe.process(gl, gr, time.time())
        if alert:
            speaker.say(alert.text)
        if a.show:
            cv2.imshow("koru", visualize(gl, info["disp"], info["levels"], cfg))
            if cv2.waitKey(1) == ord("q"):
                break


def calibrate_stereo(folder, board=(9, 6), square_m=0.025, out="calib.npz", cfg: Cfg = CFG):


    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square_m
    obj, pts_l, pts_r = [], [], []
    for lp, rp in zip(sorted(glob(f"{folder}/left_*.png")), sorted(glob(f"{folder}/right_*.png"))):
        gl, gr = (cv2.resize(cv2.imread(p, 0), (cfg.W, cfg.H)) for p in (lp, rp))
        ok_l, c_l = cv2.findChessboardCorners(gl, board)
        ok_r, c_r = cv2.findChessboardCorners(gr, board)
        if ok_l and ok_r:
            pts_l.append(c_l)
            pts_r.append(c_r)
            obj.append(objp)
    if len(obj) < 10:
        raise SystemExit(f"only {len(obj)} usable image pairs found - take at least 10-15")
    size = (cfg.W, cfg.H)
    _, K1, D1, _, _ = cv2.calibrateCamera(obj, pts_l, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj, pts_r, size, None, None)
    err, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj, pts_l, pts_r, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
    np.savez(out, w=cfg.W, h=cfg.H, K1=K1, D1=D1, R1=R1, P1=P1, K2=K2, D2=D2, R2=R2, P2=P2)
    print(f"{len(obj)} pairs used, reprojection error {err:.2f}px, "
          f"baseline {-P2[0, 3] / P2[0, 0] * 1000:.0f} mm, focal length {P2[0, 0]:.0f} px -> saved {out}")
    return err


def check_baseline(n=150):

    print(f"generating {n} synthetic stereo pairs...")
    d = build_dataset(n, 20_000_000, f"koru_check_{n}.npz")
    print("class counts (clear/caution/danger):", np.bincount(d["y"].ravel(), minlength=3))
    report("baseline", d["y"], d["y_base"])
    err = np.abs(d["near_base"] - d["near_gt"])
    both = np.isfinite(d["near_base"]) & np.isfinite(d["near_gt"]) & (d["near_gt"] < 3)
    print(f"  median distance error where both see an obstacle < 3 m: {np.median(err[both]):.2f} m")


def main():
    ap = argparse.ArgumentParser(description="Koru stereo-vision glasses prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="test simulator + rule-based detector (no TensorFlow)")
    p = sub.add_parser("train", help="generate data, train TF model, evaluate, export TFLite")
    p.add_argument("--n-train", type=int, default=1200)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--out", default="koru_model.keras")
    p = sub.add_parser("demo", help="simulated walk towards obstacles")
    p.add_argument("--model", default="koru_model.keras")
    p = sub.add_parser("calibrate", help="stereo calibration from chessboard photos")
    p.add_argument("--folder", required=True)
    p.add_argument("--out", default="calib.npz")
    p = sub.add_parser("live", help="run on two real cameras")
    p.add_argument("--left", type=int, default=0)
    p.add_argument("--right", type=int, default=1)
    p.add_argument("--calib", default=None)
    p.add_argument("--model", default="koru_model.tflite")
    p.add_argument("--show", action="store_true")
    a = ap.parse_args()

    if a.cmd == "baseline":
        check_baseline()
    elif a.cmd == "train":
        train(a.n_train, a.n_val, a.epochs, a.out)
    elif a.cmd == "demo":
        run_demo(a.model)
    elif a.cmd == "calibrate":
        calibrate_stereo(a.folder, out=a.out)
    elif a.cmd == "live":
        run_live(a)


if __name__ == "__main__":
    main()
