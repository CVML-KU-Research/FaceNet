import sys
# sys.path.append('/yolo-face')  # point to cloned folder

import os

import cv2

from typing import Dict, Any
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn



def generate_snapshot_path(base_dir: str, meta: Dict[str, Any], args, phase: str = None, add_timestamp: bool = True) -> str:
    """
    Generate a snapshot path string from data/meta and training arguments.
    Includes model type, classifier, embedding size, color mode, augmentation, phase, and timestamp for reproducibility.

    Args:
        base_dir (str): The directory to store experiment snapshots/checkpoints.
        meta (dict): Snapshot meta from data loader (e.g., class count, grayscale, aug_type).
        args: Training config or argparse/config object (must have model_name, classifier_type, embedding_size).
        phase (str, optional): Training phase (e.g., "head_only", "last_block", etc.).
        add_timestamp (bool): Whether to add a timestamp prefix for uniqueness.

    Returns:
        str: Full path to snapshot directory.
    """
    color_mode = 'gray' if meta.get('grayscale', False) else 'rgb'
    # Build main snapshot name
    parts = [
        f"model_{getattr(args, 'model_name', 'unknown')}",
        f"head_{getattr(args, 'classifier_type', 'linear')}",
        # f"embed{getattr(args, 'embedding_size', 512)}",
        # f"loss_{getattr(args, 'loss_type', 'ce')}",
        # f"batch{meta.get('batch_size', 32)}",
        # f"aug_{meta.get('aug_type', 'none')}",
        # f"{color_mode}",
        f"opt_{getattr(args, 'optimizer', 'adamW')}",
        # f"classes{meta.get('num_classes', 0)}"
    ]
    if phase:
        parts.append(f"phase_{phase}")
    # Optionally add timestamp
    if add_timestamp:
        from datetime import datetime
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts = [now] + parts
    snapshot_name = "__".join(parts)
    snapshot_path = os.path.join(base_dir, snapshot_name)
    os.makedirs(snapshot_path, exist_ok=True)
    return snapshot_path

class RetinaFacePyPIAdapter:
    def __init__(self, threshold=0.8):
        from retinaface import RetinaFace
        self.detector = RetinaFace
        self.threshold = threshold

    def detect_faces(self, frame):
        faces = self.detector.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return []
        bboxes = []
        for face_id, face_info in faces.items():
            area = face_info.get('facial_area', None)
            score = face_info.get('score', 1.0)
            if area and score >= self.threshold:
                bboxes.append(tuple(map(int, area)))
        return bboxes

class MTCNNAdapter:
    def __init__(self):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device='cuda')

    def detect_faces(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        if boxes is None: return []
        return [tuple(map(int, box)) for box in boxes]
