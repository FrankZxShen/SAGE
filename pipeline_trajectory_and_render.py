#!/usr/bin/env python3
"""
"""
import os
import sys
import json
import math
import argparse
import random
import heapq
import base64
import re
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

try:
    from scipy import ndimage as _ndi
except ImportError:
    _ndi = None

# Optional imports for Analysis pipeline
chromadb = None
YOLO = None
sv = None
openai = None

try:
    import chromadb
except ImportError:
    print("[Warning] chromadb not found. Experience storage disabled.")

try:
    from ultralytics import YOLO
except ImportError:
    print("[Warning] ultralytics not found. Object detection disabled.")

try:
    import supervision as sv
except ImportError:
    print("[Warning] supervision not found. Visualization disabled.")

try:
    import openai
except ImportError:
    print("[Warning] openai not found. VLM generation disabled.")

# ============================================================
# Analysis Helper Classes & Functions (Ported from habitat_task_exp_gen.py)
# ============================================================

class DetectionModel:
    def __init__(self, device="cuda"):
        # Default to a standard YOLO-World model or similar
        self.model = YOLO("yolov8x-world.pt") 
        self.model.to(device)
        self.classes = []

    def set_classes(self, classes):
        self.classes = classes
        self.model.set_classes(classes)

    def predict(self, image, conf=0.1, verbose=False):
        return self.model.predict(image, conf=conf, verbose=verbose)

class SimpleSceneGraph2D:
    """
    Simplified SceneGraph that only relies on 2D detections and spatial relations.
    Does NOT require Depth or 3D projection.
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.detection_model = DetectionModel(device=device)
        # Simplified classes list (can be expanded)
        self.obj_classes = [
            "chair", "table", "sofa", "bed", "plant", "lamp", "tv", "monitor", 
            "keyboard", "mouse", "sink", "toilet", "bathtub", "cabinet", 
            "refrigerator", "microwave", "oven", "door", "window", "picture"
        ]
        self.detection_model.set_classes(self.obj_classes)

    def update(self, rgb, frame_idx, img_path):
        # 1. Detection
        results = self.detection_model.predict(rgb, conf=0.5, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
        
        # 2. Filter Detections (Simple confidence threshold)
        # In original code, there was complex filtering. Here we trust YOLO conf=0.5
        
        all_2d_labels = [
            f"{self.obj_classes[class_id]} {class_idx}"
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        
        # 3. Calculate Spatial Relations (2D)
        H, W = rgb.shape[:2]
        eps = 0.05
        obj_infos_2d = []
        for i in range(len(xyxy_np)):
            curr_class_name = self.obj_classes[detection_class_ids[i]]
            x1, y1, x2, y2 = xyxy_np[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            obj_infos_2d.append((curr_class_name, float(cx / W), float(cy / H)))

        scene_graph = {}
        for i in range(len(obj_infos_2d)):
            li, xi, yi = obj_infos_2d[i]
            # Unique key for graph? Original code used class name which overwrites duplicates
            # Let's use formatted label to be unique if needed, or stick to class name as in original
            # Original: li = obj_infos_2d[i] which is (name, x, y) tuple? No, li is name.
            # Wait, original code: li, xi, yi = obj_infos_2d[i] -> li is name.
            # If multiple chairs, they overwrite each other in scene_graph[li]. 
            # This seems to be a bug/feature of the original code. We keep it for consistency.
            
            if li not in scene_graph:
                scene_graph[li] = {}
            for j in range(len(obj_infos_2d)):
                if i == j:
                    continue
                lj, xj, yj = obj_infos_2d[j]
                dx = xj - xi
                dy = yj - yi
                m = max(abs(dx), abs(dy))
                if m <= eps:
                    direction = "same"
                else:
                    if abs(dx) >= abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "down" if dy > 0 else "up"
                scene_graph[li][lj] = direction

        # Visualization
        annotated_image = rgb.copy()
        if sv:
            detections = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids
            )
            box_annotator = sv.BoundingBoxAnnotator()
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
            labels_vis = [
                f"{self.obj_classes[class_id]} {confidence:0.2f}"
                for class_id, confidence in zip(detection_class_ids, confidences)
            ]
            label_annotator = sv.LabelAnnotator()
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_vis)

        return annotated_image, all_2d_labels, scene_graph

def convert_scene_graph_to_text(scene_graph):
    descriptions = []
    for subject, relations in scene_graph.items():
        for object_name, relation in relations.items():
            if relation == "right": desc = "to the left of"
            elif relation == "left": desc = "to the right of"
            elif relation == "down": desc = "above"
            elif relation == "up": desc = "below"
            elif relation == "same": desc = "at the same position as"
            else: continue
            descriptions.append(f'"{subject}" is {desc} the "{object_name}"')

    if not descriptions:
        return ""
    
    formatted_sentences = [descriptions[0] + "."]
    for d in descriptions[1:]:
        d = d[0].upper() + d[1:] + "."
        formatted_sentences.append(d)
        
    return "The spatial relationship where " + " ".join(formatted_sentences)

# ---- Prompts (Mocked) ----
def EQA_Prompt_format(img, objects, core_relationship):
    sys_prompt = "You are an intelligent navigation agent."
    user_prompt = f"Image: [IMAGE]\nObjects: {objects}\nRelations: {core_relationship}\nGenerate a task: [Task Format] ... [Question] ... [Answer] ..."
    return sys_prompt, user_prompt

def ObjectGoal_Prompt_format(img, objects, core_relationship):
    sys_prompt = "You are an intelligent navigation agent."
    user_prompt = f"Image: [IMAGE]\nObjects: {objects}\nRelations: {core_relationship}\nDetermine the best object goal."
    return sys_prompt, user_prompt

def EXP_EQA(question, img, objects, relations):
    sys_prompt = "You are generating navigation experiences."
    user_prompt = f"Question: {question}\nImage: [IMAGE]\nContext: {objects}, {relations}\nWrite an experience following: IF answering ... AND observing ... THEN prioritize this path."
    return sys_prompt, user_prompt

def EXP_ObjectGoal(goal, img, objects, relations):
    sys_prompt = "You are generating navigation experiences."
    user_prompt = f"Goal: {goal}\nImage: [IMAGE]\nContext: {objects}, {relations}\nWrite an experience following: IF searching for ... AND observing ... THEN prioritize this path."
    return sys_prompt, user_prompt

# ---- VLM Call ----
def call_vlm(sys_prompt, user_prompt, img_base64=None):
    if not openai:
        return "VLM_DISABLED"
    
    messages = [{"role": "system", "content": sys_prompt}]
    content = []
    # Simple prompt parsing
    text_part = user_prompt.replace("[IMAGE]", "")
    content.append({"type": "text", "text": text_part})
    if img_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
        })
    messages.append({"role": "user", "content": content})
    
    try:
        # Assuming Qwen compatible API
        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        )
        completion = client.chat.completions.create(
            model="qwen-vl-max", # or whatever model
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"VLM Call Error: {e}")
        return "VLM_ERROR"

# ============================================================
# 第一部分：轨迹生成（来自 generate_virtual_trajectories_astar.py）
# ============================================================

@dataclass
class TrajConfig:
    # ---- dataset size ----
    num_trajs: int = 50
    seed: int = 0

    # ---- occupancy interpretation ----
    clearance_px: int = 2
    allow_unknown: bool = True

    # ---- extra safety margin (meters) ----
    min_clearance_m: float = 0.30

    # ---- path constraints (meters) ----
    min_path_m: float = 2.0
    max_path_m: float = 12.0
    max_pair_tries: int = 300

    # ---- graph connectivity ----
    eight_connected: bool = True
    forbid_diagonal_corner_cut: bool = True

    # ---- sampling along the shortest path ----
    num_points_min: int = 5
    num_points_max: int = 10

    # ---- camera pose ----
    cam_height_m: float = 1.5
    side_yaw_offset_deg: float = 120.0

    # ---- output ----
    save_debug_maps: bool = True
    save_free_mask_debug: bool = False
    save_dist_debug: bool = False


def load_occupancy_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)

def make_free_mask(occ: np.ndarray, clearance_px: int, allow_unknown: bool) -> np.ndarray:
    if allow_unknown:
        free = (occ == 255) | (occ == 127)
    else:
        free = (occ == 255)
    blocked = ~free
    if clearance_px > 0:
        kernel = np.ones((2 * clearance_px + 1, 2 * clearance_px + 1), dtype=bool)
        if _HAS_CV2:
            blocked_d = cv2.dilate(blocked.astype(np.uint8) * 255, kernel.astype(np.uint8), iterations=1) > 0
        elif _ndi is not None:
            blocked_d = _ndi.binary_dilation(blocked, structure=kernel, iterations=1)
        else:
            raise ImportError("Please install `opencv-python` or `scipy` for dilation/distance transform.")
        free = ~blocked_d
    return free

def clamp_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def yaw_from_uv(u0: int, v0: int, u1: int, v1: int) -> float:
    dx = float(u1 - u0)
    dy = float(v0 - v1)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.atan2(dx, dy)

def save_free_mask(free_mask: np.ndarray, out_path: str) -> None:
    img = (free_mask.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(out_path)

def save_dist_debug(dist_px: np.ndarray, out_path: str) -> None:
    d = dist_px.copy().astype(np.float32)
    d = np.clip(d, 0, np.percentile(d, 99))
    d = (d / (d.max() + 1e-6) * 255).astype(np.uint8)
    Image.fromarray(d, mode="L").save(out_path)

def pixel_to_world(u: float, v: float, W: int, H: int, meta: Dict[str, Any], z: float) -> Tuple[float, float, float]:
    s = float(meta["scale"])
    cx, cy = float(meta["center"][0]), float(meta["center"][1])
    x = cx + (u - (W / 2.0)) * s
    y = cy - (v - (H / 2.0)) * s
    return (x, y, z)

def compute_dist_to_obstacle_px(free_mask: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        img = free_mask.astype(np.uint8)
        img = (img > 0).astype(np.uint8)
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        return dist
    if _ndi is None:
        raise ImportError("Please install `opencv-python` or `scipy` for dilation/distance transform.")
    dist = _ndi.distance_transform_edt(free_mask.astype(np.uint8))
    return dist.astype(np.float32)

def heuristic(u: int, v: int, tu: int, tv: int) -> float:
    return math.hypot(tu - u, tv - v)

def neighbors(u: int, v: int, free_mask: np.ndarray, dist_px: np.ndarray, min_clear_px: int, cfg: TrajConfig):
    H, W = free_mask.shape
    if cfg.eight_connected:
        moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
        ]
    else:
        moves = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0)]

    for du, dv, cost in moves:
        nu, nv = u + du, v + dv
        if not (0 <= nu < W and 0 <= nv < H):
            continue
        if not free_mask[nv, nu]:
            continue
        if dist_px[nv, nu] < min_clear_px:
            continue
        if cfg.eight_connected and cfg.forbid_diagonal_corner_cut and abs(du) == 1 and abs(dv) == 1:
            if not (free_mask[v, nu] and free_mask[nv, u]):
                continue
            if dist_px[v, nu] < min_clear_px or dist_px[nv, u] < min_clear_px:
                continue
        yield nu, nv, cost

def astar_path(
    free_mask: np.ndarray,
    dist_px: np.ndarray,
    min_clear_px: int,
    start: Tuple[int,int],
    goal: Tuple[int,int],
    cfg: TrajConfig
) -> Optional[List[Tuple[int,int]]]:
    su, sv = start
    gu, gv = goal
    if (su, sv) == (gu, gv):
        return [(su, sv)]

    if dist_px[sv, su] < min_clear_px or dist_px[gv, gu] < min_clear_px:
        return None

    open_heap = []
    heapq.heappush(open_heap, (heuristic(su, sv, gu, gv), 0.0, (su, sv)))

    came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
    gscore: Dict[Tuple[int,int], float] = {(su, sv): 0.0}
    closed = set()

    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)

        if cur == (gu, gv):
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        cu, cv = cur
        for nu, nv, cost in neighbors(cu, cv, free_mask, dist_px, min_clear_px, cfg):
            ng = g + cost
            if (nu, nv) not in gscore or ng < gscore[(nu, nv)]:
                gscore[(nu, nv)] = ng
                came_from[(nu, nv)] = (cu, cv)
                nf = ng + heuristic(nu, nv, gu, gv)
                heapq.heappush(open_heap, (nf, ng, (nu, nv)))

    return None

def path_geodesic_px(path: List[Tuple[int,int]]) -> float:
    if len(path) < 2:
        return 0.0
    s = 0.0
    for (u0,v0),(u1,v1) in zip(path[:-1], path[1:]):
        s += math.hypot(u1-u0, v1-v0)
    return s

def sample_equal_distance_points(path: List[Tuple[int,int]], num_points: int) -> List[Tuple[int,int]]:
    assert num_points >= 2
    if len(path) == 1:
        return [path[0]] * num_points

    pts = np.array(path, dtype=np.float32)
    seg = pts[1:] - pts[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cum[-1])

    if total < 1e-6:
        return [tuple(map(int, path[0]))] * num_points

    targets = np.linspace(0.0, total, num_points)

    sampled = []
    j = 0
    for t in targets:
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            sampled.append(tuple(map(int, path[-1])))
            continue
        t0, t1 = cum[j], cum[j + 1]
        alpha = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        p = pts[j] * (1 - alpha) + pts[j + 1] * alpha
        sampled.append((int(round(p[0])), int(round(p[1]))))

    cleaned = [sampled[0]]
    for p in sampled[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    while len(cleaned) < num_points:
        cleaned.append(cleaned[-1])

    return cleaned[:num_points]

def draw_path_on_occ(occ: np.ndarray, path_px: List[Tuple[int,int]], sampled_px: List[Tuple[int,int]], out_path: str) -> None:
    if _HAS_CV2:
        img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

        if len(path_px) >= 2:
            pts = np.array(path_px, dtype=np.int32)
            for i in range(1, len(pts)):
                cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), (255, 0, 0), 1, cv2.LINE_AA)

        if len(sampled_px) >= 2:
            spts = np.array(sampled_px, dtype=np.int32)
            for i in range(1, len(spts)):
                cv2.line(img, tuple(spts[i - 1]), tuple(spts[i]), (0, 0, 255), 2, cv2.LINE_AA)
            for p in spts:
                cv2.circle(img, tuple(p), 3, (0, 255, 255), -1)
            cv2.circle(img, tuple(spts[0]), 5, (0, 255, 0), -1)
            cv2.circle(img, tuple(spts[-1]), 5, (0, 0, 0), -1)

        cv2.imwrite(out_path, img)
        return

    from PIL import ImageDraw

    img = Image.fromarray(occ, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    if len(path_px) >= 2:
        draw.line(path_px, fill=(0, 0, 255), width=1)

    if len(sampled_px) >= 2:
        draw.line(sampled_px, fill=(255, 0, 0), width=2)
        for (u, v) in sampled_px:
            r = 3
            draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 255, 0))
        (u0, v0) = sampled_px[0]
        (u1, v1) = sampled_px[-1]
        r0 = 5
        draw.ellipse((u0 - r0, v0 - r0, u0 + r0, v0 + r0), fill=(0, 255, 0))
        draw.ellipse((u1 - r0, v1 - r0, u1 + r0, v1 + r0), fill=(0, 0, 0))

    img.save(out_path)

def export_poses_jsonl_from_sampled(
    sampled_px: List[Tuple[int,int]],
    occ_meta: Dict[str, Any],
    occ_W: int,
    occ_H: int,
    cfg: TrajConfig,
    out_jsonl: str,
) -> None:
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    side_yaw_offset = math.radians(float(cfg.side_yaw_offset_deg))
    yaws = []
    for i in range(len(sampled_px)):
        if i < len(sampled_px) - 1:
            (u0,v0) = sampled_px[i]
            (u1,v1) = sampled_px[i+1]
        else:
            (u0,v0) = sampled_px[i-1]
            (u1,v1) = sampled_px[i]
        yaws.append(clamp_angle(yaw_from_uv(u0,v0,u1,v1)))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t, ((u, v), yaw) in enumerate(zip(sampled_px, yaws)):
            x, y, z = pixel_to_world(u, v, occ_W, occ_H, occ_meta, cfg.cam_height_m)
            rec = {
                "t": int(t),
                "pixel_uv": [int(u), int(v)],
                "views": {
                    "front": {"position": [x, y, z], "yaw": float(yaw)},
                    "left":  {"position": [x, y, z], "yaw": float(clamp_angle(yaw - side_yaw_offset))},
                    "right": {"position": [x, y, z], "yaw": float(clamp_angle(yaw + side_yaw_offset))},
                },
            }
            f.write(json.dumps(rec) + "\n")

def choose_free_pixel_safe(free_mask: np.ndarray, dist_px: np.ndarray, min_clear_px: int) -> Tuple[int, int]:
    ys, xs = np.where(free_mask & (dist_px >= float(min_clear_px)))
    if len(xs) == 0:
        raise ValueError("No safe free space found (free & dist>=min_clear_px). Try reducing min_clearance_m or clearance_px.")
    i = random.randrange(len(xs))
    return int(xs[i]), int(ys[i])

def sample_one_astar_traj(
    free_mask: np.ndarray,
    dist_px: np.ndarray,
    min_clear_px: int,
    occ_scale_m_per_px: float,
    cfg: TrajConfig
) -> Optional[Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]]:
    for _ in range(cfg.max_pair_tries):
        S = choose_free_pixel_safe(free_mask, dist_px, min_clear_px)
        T = choose_free_pixel_safe(free_mask, dist_px, min_clear_px)

        du = T[0] - S[0]
        dv = T[1] - S[1]
        euclid_m = math.hypot(du, dv) * occ_scale_m_per_px
        if euclid_m < cfg.min_path_m * 0.6 or euclid_m > cfg.max_path_m * 1.2:
            continue

        path = astar_path(free_mask, dist_px, min_clear_px, S, T, cfg)
        if path is None or len(path) < 2:
            continue

        geo_m = path_geodesic_px(path) * occ_scale_m_per_px
        if not (cfg.min_path_m <= geo_m <= cfg.max_path_m):
            continue

        N = random.randint(cfg.num_points_min, cfg.num_points_max)
        sampled = sample_equal_distance_points(path, N)
        if len(sampled) < 2:
            continue

        ok = True
        for (u, v) in sampled:
            if not free_mask[v, u] or dist_px[v, u] < min_clear_px:
                ok = False
                break
        if not ok:
            continue

        return path, sampled

    return None


# ============================================================
# 第二部分：3DGS 渲染（来自 render_from_poses.py）
# ============================================================

# 添加 3DGS_PoseRender 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "3DGS_PoseRender"))

try:
    from gaussian_model import GaussianModel
    from camera import Camera
    from render import render
except ImportError as e:
    print(f"[警告] 无法导入3DGS渲染模块: {e}")
    print("       如果只需要生成轨迹，可以使用 --skip-render 参数")
    GaussianModel = None
    Camera = None
    render = None

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)

def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def yaw_pitch_to_c2w(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    f = _normalize(np.array([sy * cp, -cy * cp, sp], dtype=np.float32))

    r = np.cross(WORLD_UP, f)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), f)
    r = _normalize(r)

    u = _normalize(np.cross(f, r))
    down = -u

    R_c2w = np.stack([r, down, f], axis=1).astype(np.float32)

    if np.linalg.det(R_c2w) < 0:
        R_c2w[:, 0] *= -1.0

    return R_c2w

def save_rgb(rgb_tensor, out_path):
    import torch
    if isinstance(rgb_tensor, torch.Tensor):
        rgb_np = rgb_tensor.detach().cpu().numpy()
        if rgb_np.shape[0] == 3:
            rgb_np = np.transpose(rgb_np, (1, 2, 0))
    else:
        rgb_np = rgb_tensor
    
    img = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


# ============================================================
# Pipeline 主函数
# ============================================================

def generate_trajectories(scene_dir: str, out_root: str, cfg: TrajConfig) -> int:
    """生成轨迹，返回成功生成的轨迹数量"""
    print("\n" + "="*60)
    print("步骤 1/2: 生成虚拟轨迹")
    print("="*60)
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    occ_png = os.path.join(scene_dir, "occupancy.png")
    occ_json = os.path.join(scene_dir, "occupancy.json")

    occ = load_occupancy_png(occ_png)
    H, W = occ.shape
    with open(occ_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    scale = float(meta["scale"])

    free_mask = make_free_mask(occ, cfg.clearance_px, cfg.allow_unknown)
    dist_px = compute_dist_to_obstacle_px(free_mask)
    min_clear_px = int(math.ceil(cfg.min_clearance_m / scale))
    min_clear_px = max(1, min_clear_px)

    os.makedirs(out_root, exist_ok=True)
    if cfg.save_free_mask_debug:
        save_free_mask(free_mask, os.path.join(out_root, "free_mask.png"))
    if cfg.save_dist_debug:
        save_dist_debug(dist_px, os.path.join(out_root, "dist_to_obstacle.png"))

    wrote = 0
    guard = 0
    while wrote < cfg.num_trajs and guard < cfg.num_trajs * 80:
        guard += 1
        res = sample_one_astar_traj(free_mask, dist_px, min_clear_px, scale, cfg)
        if res is None:
            continue
        full_path, sampled = res

        traj_dir = os.path.join(out_root, f"traj_{wrote:04d}")
        os.makedirs(traj_dir, exist_ok=True)

        export_poses_jsonl_from_sampled(sampled, meta, W, H, cfg, os.path.join(traj_dir, "poses.jsonl"))

        if cfg.save_debug_maps:
            draw_path_on_occ(occ, full_path, sampled, os.path.join(traj_dir, "topdown_path.png"))

        wrote += 1
        if (wrote) % 10 == 0 or wrote == cfg.num_trajs:
            print(f"  已生成 {wrote}/{cfg.num_trajs} 条轨迹...")

    if wrote < cfg.num_trajs:
        print(f"[警告] 仅生成了 {wrote}/{cfg.num_trajs} 条轨迹")
        print("       建议: 降低 min_clearance_m、clearance_px 或增加 max_pair_tries")
    
    print(f"✓ 轨迹生成完成: {wrote} 条轨迹 -> {out_root}")
    print(f"  最小障碍物间隙: {cfg.min_clearance_m:.3f} m (~{min_clear_px} px @ scale={scale})")
    
    return wrote


def render_trajectories(
    ply_path: str,
    traj_root: str,
    out_root: str,
    width: int,
    height: int,
    hfov_deg: float,
    pitch_deg: float,
    views: List[str],
    max_frames: int,
    skip_analysis: bool = False,
) -> None:
    """渲染所有轨迹并执行场景分析与经验生成"""
    print("\n" + "="*60)
    print("步骤 2/2: 渲染 3DGS 图像" + (" & 生成导航经验" if not skip_analysis else ""))
    print("="*60)
    
    if GaussianModel is None:
        print("[错误] 3DGS渲染模块未加载，无法执行渲染")
        return

    # 加载高斯模型
    print(f"  加载3DGS模型: {ply_path}")
    gaussians = GaussianModel()
    gaussians.load(ply_path)

    # 计算焦距
    hfov_rad = math.radians(hfov_deg)
    fx = fy = width / (2 * math.tan(hfov_rad / 2))

    # 初始化分析工具
    scene_graph = None
    if not skip_analysis and YOLO:
        print("  初始化 SceneGraph (YOLO-World)...")
        scene_graph = SimpleSceneGraph2D()
    elif not skip_analysis and not YOLO:
        print("  [Warning] YOLO not found, analysis will be skipped.")
    
    chroma_client = None
    collection = None
    if not skip_analysis and chromadb:
        chroma_db_path = os.path.join(out_root, "chroma_db")
        if not os.path.exists(chroma_db_path):
            os.makedirs(chroma_db_path)
        print(f"  初始化 ChromaDB: {chroma_db_path}")
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        collection = chroma_client.get_or_create_collection(name="nav_experiences")

    # 遍历所有轨迹
    traj_names = sorted([d for d in os.listdir(traj_root) if os.path.isdir(os.path.join(traj_root, d))])
    total_trajs = len(traj_names)
    
    for idx, traj_name in enumerate(traj_names, 1):
        traj_dir = os.path.join(traj_root, traj_name)
        poses_path = os.path.join(traj_dir, "poses.jsonl")
        if not os.path.isfile(poses_path):
            continue

        out_traj_dir = os.path.join(out_root, traj_name)
        
        # 读取所有点
        points = []
        with open(poses_path, "r", encoding="utf-8") as f:
            for line in f:
                points.append(json.loads(line))
        
        if max_frames > 0:
            points = points[:max_frames]

        seed_data = []
        path_valid = True
        
        # 逐点处理
        # 强制处理顺序: front, left, right (为了检查 validity)
        ordered_views = ["front", "left", "right"]
        
        print(f"  [{idx}/{total_trajs}] 处理 {traj_name} ({len(points)} points)...")
        
        for i, rec in enumerate(points):
            t = int(rec["t"])
            point_data = {
                "point_idx": t,
                "point_coord": rec["views"]["front"]["position"],
                "views": []
            }
            
            point_dir = os.path.join(out_traj_dir, f"point_{t}")
            os.makedirs(point_dir, exist_ok=True)
            
            for view_name in ordered_views:
                if view_name not in views:
                    continue
                    
                # 渲染
                pos = np.array(rec["views"][view_name]["position"], dtype=np.float32)
                yaw = float(rec["views"][view_name]["yaw"])
                R = yaw_pitch_to_c2w(yaw, math.radians(pitch_deg))
                
                cam_info = {
                    "position": pos,
                    "rotation": R,
                    "fx": fx, "fy": fy,
                    "width": width, "height": height,
                }
                cam = Camera()
                cam.load(cam_info)
                
                rgb_tensor = render(cam, gaussians)
                
                # 保存图像
                img_filename = f"point_{t}_{view_name}.png"
                save_path = os.path.join(point_dir, img_filename)
                save_rgb(rgb_tensor, save_path)
                
                # 场景分析
                all_2d_labels = []
                scene_graph_text = ""
                if scene_graph:
                    # Convert tensor to numpy (H, W, 3) uint8
                    rgb_np = rgb_tensor.detach().cpu().numpy().transpose(1, 2, 0)
                    rgb_np = (np.clip(rgb_np, 0, 1) * 255).astype(np.uint8)
                    
                    annotated_img, all_2d_labels, sg_dict = scene_graph.update(rgb_np, t, save_path)
                    scene_graph_text = convert_scene_graph_to_text(sg_dict)
                    
                    # Optional: Save annotated image
                    # ann_save_path = os.path.join(point_dir, f"point_{t}_{view_name}_ann.png")
                    # Image.fromarray(annotated_img).save(ann_save_path)

                # Validity Check (只检查最后一个点的 front 视图)
                if i == len(points) - 1 and view_name == "front":
                    # 如果跳过分析，默认路径有效
                    if skip_analysis:
                        path_valid = True
                    # 否则检查是否有检测结果
                    elif not all_2d_labels:
                        path_valid = False
                        print(f"    [Warning] Path marked as invalid: No objects detected at last point (front view). Labels: {all_2d_labels}")
                    else:
                        print(f"    [Info] Path valid. Last point detected: {all_2d_labels}")
                    # 可以在此添加 'door', 'power outlet' 等排除逻辑
                
                if not path_valid:
                    # break # 暂时注释掉 break，以便观察后续视角
                    pass 
                
                view_data = {
                    "view": view_name,
                    "filename": img_filename,
                    "all_2d_labels": all_2d_labels,
                    "scene_graph_text": scene_graph_text
                }
                point_data["views"].append(view_data)
            
            # if not path_valid:
            #    break
            
            seed_data.append(point_data)
            
        if not path_valid:
            print(f"    [Skip] Path marked as invalid (no targets at end). BUT KEEPING FILES FOR DEBUG: {out_traj_dir}")
            # if os.path.exists(out_traj_dir):
            #     shutil.rmtree(out_traj_dir)
            # continue
            pass # 暂时保留文件
            
        # 生成经验 (VLM) 并写入 ChromaDB
        if not skip_analysis and openai and collection and seed_data:
            try:
                # 获取最后一个点的 Front 视图信息
                last_point = seed_data[-1]
                last_front_view = next((v for v in last_point["views"] if v["view"] == "front"), None)
                
                if last_front_view:
                    # Read Last Image
                    last_img_path = os.path.join(out_traj_dir, f"point_{last_point['point_idx']}", last_front_view["filename"])
                    with open(last_img_path, "rb") as f:
                        last_img_b64 = base64.b64encode(f.read()).decode('utf-8')
                    
                    # 1. Generate EQA Question
                    sys_p, user_p = EQA_Prompt_format(
                        last_img_b64, last_front_view["all_2d_labels"], last_front_view["scene_graph_text"]
                    )
                    eqa_raw = call_vlm(sys_p, user_p, last_img_b64)
                    
                    # Parse EQA (Simple regex)
                    task_format, eqa_q, eqa_a = "Unknown", "Unknown", "Unknown"
                    match = re.search(r"Task Format:\s*(.*?)\s*Question:\s*(.*?)\s*Answer:\s*(.*)", eqa_raw, re.IGNORECASE | re.DOTALL)
                    if match:
                        task_format, eqa_q, eqa_a = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                    
                    # 2. Generate Object Goal
                    sys_p, user_p = ObjectGoal_Prompt_format(
                        last_img_b64, last_front_view["all_2d_labels"], last_front_view["scene_graph_text"]
                    )
                    object_goal = call_vlm(sys_p, user_p, last_img_b64)
                    
                    print(f"    Generated Task: {eqa_q} | Goal: {object_goal}")
                    
                    # 3. Generate Experiences for ALL points
                    for p_idx, p_data in enumerate(seed_data):
                        # Use Front view for experience generation
                        p_front = next((v for v in p_data["views"] if v["view"] == "front"), None)
                        if not p_front: continue
                        
                        # Read point image
                        p_img_path = os.path.join(out_traj_dir, f"point_{p_data['point_idx']}", p_front["filename"])
                        with open(p_img_path, "rb") as f:
                            p_img_b64 = base64.b64encode(f.read()).decode('utf-8')
                            
                        # Exp: EQA
                        sys_p, user_p = EXP_EQA(eqa_q, p_img_b64, p_front["all_2d_labels"], p_front["scene_graph_text"])
                        eqa_exp = call_vlm(sys_p, user_p, p_img_b64)
                        
                        # Exp: ObjGoal
                        sys_p, user_p = EXP_ObjectGoal(object_goal, p_img_b64, p_front["all_2d_labels"], p_front["scene_graph_text"])
                        obj_exp = call_vlm(sys_p, user_p, p_img_b64)
                        
                        p_data["eqa_experience"] = eqa_exp
                        p_data["objectgoal_experience"] = obj_exp
                        
                        # Write to Chroma
                        base_id = f"{traj_name}_{p_idx}"
                        collection.add(
                            documents=[eqa_exp, obj_exp],
                            metadatas=[
                                {"type": "eqa", "traj": traj_name, "point": p_idx, "q": eqa_q},
                                {"type": "obj_goal", "traj": traj_name, "point": p_idx, "goal": object_goal}
                            ],
                            ids=[f"{base_id}_eqa", f"{base_id}_obj"]
                        )
                    
                    # Save JSON
                    json_data = {
                        "traj_name": traj_name,
                        "task_format": task_format,
                        "eqa_q": eqa_q,
                        "eqa_a": eqa_a,
                        "object_goal": object_goal,
                        "points": seed_data
                    }
                    with open(os.path.join(out_traj_dir, f"{traj_name}.json"), "w") as f:
                        json.dump(json_data, f, indent=2)
                        
            except Exception as e:
                print(f"    [Error] Analysis failed for {traj_name}: {e}")

    print(f"✓ 处理完成: {out_root}")


