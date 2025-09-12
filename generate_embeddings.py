
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from natsort import natsorted

def extract_embeddings(backbone, data_dir, device, crop_size=(112,112), grayscale=False, batch_size=32, out_path="face_gallery.npz"):
    """
    Extracts and saves all embeddings from a folder structured as data_dir/class_name/image.jpg.
    Saves to .npz: embeddings (N, D), labels (N), paths (N).
    """
    tr_list = []
    if grayscale:
        tr_list.append(transforms.Grayscale(num_output_channels=3))
    tr_list.extend([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.31928780674934387, 0.2873991131782532, 0.25779902935028076],
            std=[0.19799138605594635, 0.20757903158664703, 0.21088403463363647]
        )
    ])
    preprocess = transforms.Compose(tr_list)
    img_paths, labels = [], []
    class_names = sorted(os.listdir(data_dir))  # Ensure consistent order
    # print(f"Found {len(class_names)} classes: {class_names}")

    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('jpg','png','jpeg','bmp')):
                img_paths.append(os.path.join(cls_dir, fname))
                labels.append(label)
    print(f"Found {len(img_paths)} images in total.")
    backbone.eval()
    backbone.to(device)
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(img_paths), batch_size), desc="Extracting gallery embeddings"):
            batch_paths = img_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                im = Image.open(p).convert('RGB')
                imgs.append(preprocess(im))
            imgs = torch.stack(imgs).to(device)
            embs = backbone(imgs)
            all_embs.append(embs.cpu())
    all_embs = torch.cat(all_embs, dim=0).numpy()
    labels = np.array(labels)
    img_paths = np.array(img_paths)
    # Save as .npz (fast, portable, readable)
    np.savez_compressed(out_path, embeddings=all_embs, labels=labels, paths=img_paths)
    print(f"Saved embeddings to {out_path} ({all_embs.shape[0]} faces).")

def search_gallery(emb_query, emb_gallery, labels_gallery, topk=1, threshold=None):
    """
    Find top-k most similar gallery faces for a query embedding.

    Args:
        emb_query: (1, D) numpy array
        emb_gallery: (N, D) numpy array
        labels_gallery: (N,) array of int (label indices)
        topk: int
        threshold: float or None; only accept matches above this similarity

    Returns:
        List of (label, similarity) tuples for top-k matches (or just one if topk=1)
    """
    import torch.nn.functional as F
    q = torch.from_numpy(emb_query)
    G = torch.from_numpy(emb_gallery)
    sim = F.cosine_similarity(q, G, dim=1).numpy()  # (N,)
    topk_idx = np.argsort(-sim)[:topk]
    results = []
    for idx in topk_idx:
        if threshold is None or sim[idx] >= threshold:
            results.append((labels_gallery[idx], sim[idx]))
        else:
            results.append(('Unknown', sim[idx]))
    return results

def recognize_unlabeled_faces_image(
    backbone,
    gallery_npz,         # Path to gallery .npz (produced by extract_embeddings)
    face_detector,       # Your detector object, e.g., RetinaFacePyPIAdapter
    image_path=None,     # Path to test image (or use image_array)
    image_array=None,    # Alternatively, BGR np.array (OpenCV format)
    device='cuda',
    crop_size=(112, 112),
    grayscale=False,
    mean=None, std=None,
    topk=1,
    threshold=None,      # Only report matches above this similarity
    show=True,
    save_path=None,
):
    """
    Recognize faces in an unlabeled image using gallery embeddings.

    Args:
        backbone: trained backbone model.
        gallery_npz: path to npz file with keys: 'embeddings', 'labels', 'paths'
        face_detector: object with .detect_faces()
        image_path/image_array: BGR image as path or numpy array.
        topk: int, how many closest classes to show.
        threshold: float, min cosine similarity to accept as a match.
    """

    # Load gallery
    gallery = np.load(gallery_npz)
    emb_gallery = gallery['embeddings']   # (N, D)
    labels_gallery = gallery['labels']
    paths_gallery = gallery['paths']
    print(f"Loaded gallery: {emb_gallery.shape[0]} faces, {emb_gallery.shape[1]}D embeddings")

    # Get class_names via folder names (for label mapping)
    class_names = natsorted(list(set([os.path.basename(os.path.dirname(p)) for p in paths_gallery])))

    # Prepare image
    if image_array is not None:
        img = image_array.copy()
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
    orig_img = img.copy()

    # Detect faces
    face_bboxes = face_detector.detect_faces(img)

    # Preprocessing transform
    tr = []
    if grayscale:
        tr.append(transforms.Grayscale(num_output_channels=3))
    tr.extend([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.302765429019928, 0.2525686025619507, 0.21395820379257202],
            std=[0.1909194439649582, 0.1914139688014984, 0.19042560458183289]
        )
    ])
    preprocess = transforms.Compose(tr)
    backbone.eval()
    backbone.to(device)

    for bbox in face_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        face_img = img[y1:y2, x1:x2]
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            emb_query = backbone(face_tensor).cpu().numpy().reshape(1, -1)

        # === Use search_gallery to get best matches ===
        match_results = search_gallery(
            emb_query=emb_query,
            emb_gallery=emb_gallery,
            labels_gallery=labels_gallery,
            topk=topk,
            threshold=threshold
        )
        # Take the best match for annotation
        match_label, match_sim = match_results[0]
        if match_label == 'Unknown':
            label_str = "Unknown"
        else:
            match_class = class_names[match_label]
            label_str = f"{match_class} ({match_sim:.2f})"
        # Draw box/label
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(orig_img, label_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Show/save
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    if save_path is not None:
        cv2.imwrite(save_path, orig_img)
        print(f"Saved result to {save_path}")
    return orig_img


def recognize_unlabeled_faces_video(
    backbone,
    gallery_npz,
    face_detector,
    video_path=None,         # Path to video file, or int (for webcam)
    device='cuda',
    crop_size=(112, 112),
    grayscale=False,
    topk=1,
    threshold=None,
    show=True,
    save_path=None,
    every_nth=1,             # Process every nth frame for speedup
    max_frames=None,         # Optionally, stop after N frames
    search_gallery=None      # Function handle for gallery search
):
    """
    Recognize faces in all frames of a video using gallery embeddings.
    """
    assert search_gallery is not None, "You must provide search_gallery function!"
    # Load gallery
    gallery = np.load(gallery_npz)
    emb_gallery = gallery['embeddings']
    labels_gallery = gallery['labels']
    paths_gallery = gallery['paths']
    class_names = natsorted(list(set([os.path.basename(os.path.dirname(p)) for p in paths_gallery])))
    # class_names = gallery['class_names']  # Use saved class names directly

    # Preprocessing pipeline
    tr = []
    if grayscale:
        tr.append(transforms.Grayscale(num_output_channels=3))
    tr.extend([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.302765429019928, 0.2525686025619507, 0.21395820379257202],
            std=[0.1909194439649582, 0.1914139688014984, 0.19042560458183289]
        )
    ])
    preprocess = transforms.Compose(tr)
    backbone.eval()
    backbone.to(device)

    # Video input/output setup
    cap = cv2.VideoCapture(video_path if video_path is not None else 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1
    print(f"[INFO] Video: {video_path}, {W}x{H}, {fps} FPS, {total_frames} frames")

    out_vid = None
    if save_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    frame_idx = 0
    processed = 0
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % every_nth != 0:
            continue
        orig_img = img.copy()
        # Detect faces
        face_bboxes = face_detector.detect_faces(img)
        # Per-face recognition
        for bbox in face_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            face_img = img[y1:y2, x1:x2]
            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = preprocess(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                emb_query = backbone(face_tensor).cpu().numpy().reshape(1, -1)
            # Find best match
            match_results = search_gallery(
                emb_query=emb_query,
                emb_gallery=emb_gallery,
                labels_gallery=labels_gallery,
                topk=topk,
                threshold=threshold
            )
            match_label, match_sim = match_results[0]
            if match_label == 'Unknown':
                label_str = "Unknown"
            else:
                match_class = class_names[match_label]
                label_str = f"{match_class} ({match_sim:.2f})"
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(orig_img, label_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # Output/Display
        if show:
            cv2.imshow('Face Recognition', orig_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if out_vid is not None:
            out_vid.write(orig_img)
        processed += 1
        if max_frames is not None and processed >= max_frames:
            break
    cap.release()
    if out_vid is not None:
        out_vid.release()
        print(f"[INFO] Saved annotated video to {save_path}")
    if show:
        cv2.destroyAllWindows()
    print(f"[INFO] Processed {processed} frames.")
