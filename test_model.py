import argparse
import os
import torch
from matplotlib import pyplot as plt

from data_loader import get_data_loader
from utils.utils import RetinaFacePyPIAdapter
from networks.models_facenet import Backbone, MobileFaceNet, Arcface, CosFace 
from generate_embeddings import extract_embeddings, recognize_unlabeled_faces_image, recognize_unlabeled_faces_video, search_gallery

import warnings
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device if needed

def get_argparser(use=False):
    parser = argparse.ArgumentParser(description="Face Recognition Training Configurations")

    parser.add_argument('--root_dir', type=str, default="./data", help='Root directory for data')
    parser.add_argument('--grayscale', action='store_true', help='Use grayscale images (default: False)')
    parser.add_argument('--aug_type', type=str, default='strong', choices=['standard', 'strong', 'none'], help='Augmentation type')
    parser.add_argument('--loss_type', type=str, default='smooth_ce', choices=['ce', 'smooth_ce'], help='Loss function')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[112, 112], help='Image crop size (h, w)')
    parser.add_argument('--model_name', type=str, default='mobilefacenet', choices=['mobilefacenet', 'ir_se', 'mobilenetv2'], help='Model backbone')
    parser.add_argument('--classifier_type', type=str, default='FC', choices=['arcface', 'FC', 'cosface', 'combined'], help='Classifier head type')
    parser.add_argument('--embedding_size', type=int, default=512, help='Embedding size')

    parser.add_argument('--checkpoints_path', type=str, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--pretrained_backbone_path', type=str, default="models/mobile_weights/model_mobilefacenet.pth", help='Pretrained backbone path')
    parser.add_argument('--pretrained_head_path', type=str, default=None, help='Pretrained classifier/head path')

    parser.add_argument('--train_batch_size', type=int, default=16, help='Train batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
    parser.add_argument('--target_acc', type=float, default=95.0, help='Target accuracy for switching phase')

    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'adamw'], help='Optimizer')
    parser.add_argument('--max_epoch', type=int, default=500, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=10, help='LR step')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='LR decay')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    parser.add_argument('--save_plot', action='store_true', help='Save training plots')
    parser.add_argument('--phase', type=str, default='head_only', choices=['head_only', 'last_block'], help='Training phase')
    parser.add_argument('--random_seed', type=int, default=2020, help='Random seed')

    if use:
        return parser.parse_args([])
    else:
        return parser.parse_args()

def load_model_head_only(args, num_classes, snapshot_path=None, device='cuda'):
    """
    Load backbone, classifier head, and weights from snapshot.
    Returns backbone, classifier, and loaded epoch.
    """
    # --- Select backbone ---
    if args.model_name.lower() == "mobilefacenet":
        backbone = MobileFaceNet(embedding_size=args.embedding_size).to(device)
    else:
        backbone = Backbone(num_layers=50, drop_ratio=0.4, mode='ir_se').to(device)

    # --- Select classifier ---
    if args.classifier_type.lower() == 'arcface':
        classifier = Arcface(embedding_size=args.embedding_size, classnum=num_classes).to(device)
    elif args.classifier_type.lower() == 'cosface':
        classifier = CosFace(embedding_size=args.embedding_size, classnum=num_classes, s=30, m=0.4).to(device)
    elif args.classifier_type.upper() == 'FC':
        classifier = torch.nn.Linear(args.embedding_size, num_classes).to(device)
    else:
        raise ValueError(f"Unknown classifier type: {args.classifier_type}")

    # --- Load best checkpoint ---
    if snapshot_path is None:
        raise ValueError("snapshot_path must be provided.")

    # Find checkpoint (by convention: contains 'best_model_epoch')
    candidates = [f for f in os.listdir(snapshot_path) if 'best_model_epoch' in f]

    if not candidates:
        raise RuntimeError(f"No best_model_epoch checkpoint found in {snapshot_path}")
    checkpoint_path = os.path.join(snapshot_path, sorted(candidates)[-1])
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    backbone.load_state_dict(checkpoint['model_state_dict'])
    classifier.load_state_dict(checkpoint['arcface_state_dict'])
    backbone.eval()
    classifier.eval()
    return backbone, classifier, checkpoint.get('epoch', None)

def load_model_last_block(args, num_classes, phase='head_only', snapshot_path=None, device='cuda'):
    """
    Load backbone, classifier head, and weights from snapshot.
    Sets proper requires_grad flags for backbone/classifier according to phase.
    Returns backbone, classifier, and loaded epoch.
    """
    # --- Select backbone ---
    if args.model_name.lower() == "mobilefacenet":
        backbone = MobileFaceNet(embedding_size=args.embedding_size).to(device)
        backbone_last_block = [
            backbone.conv_5, backbone.conv_6_sep, backbone.conv_6_dw,
            backbone.conv_6_flatten, backbone.linear, backbone.bn
        ]
    else:
        backbone = Backbone(num_layers=50, drop_ratio=0.4, mode='ir_se').to(device)
        backbone_last_block = [backbone.body[-1]]

    # --- Select classifier head ---
    if args.classifier_type.lower() == 'arcface':
        classifier = Arcface(
            embedding_size=args.embedding_size,
            classnum=num_classes,
            s=32,  # You may want to read these from args if tunable!
            m=0.4
        ).to(device)
    if args.classifier_type.lower() == 'cosface':
        classifier = CosFace(
            embedding_size=args.embedding_size,
            classnum=num_classes,
            s=30,
            m=0.4
        ).to(device)
    if args.classifier_type.lower() == 'fc':
        classifier = torch.nn.Linear(args.embedding_size, num_classes).to(device)
    # else:
    #     raise ValueError(f"Unknown classifier type: {args.classifier_type}")

    # --- Load best checkpoint ---
    if snapshot_path is None:
        raise ValueError("snapshot_path must be provided.")
    # Find checkpoint (by convention: contains 'best_model_epoch')
    candidates = [f for f in os.listdir(snapshot_path) if 'best_model_epoch' in f]
    if not candidates:
        raise RuntimeError(f"No best_model_epoch checkpoint found in {snapshot_path}")
    checkpoint_path = os.path.join(snapshot_path, sorted(candidates)[-1])
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    # classifier.load_state_dict(checkpoint['arcface_state_dict'])
    backbone.eval()
    # classifier.eval()

    # ---- Set requires_grad according to phase ----
    def set_requires_grad(module, requires_grad=True):
        for param in module.parameters():
            param.requires_grad = requires_grad

    if phase == "head_only":
        set_requires_grad(backbone, False)
        set_requires_grad(classifier, True)
    elif phase == "last_block":
        set_requires_grad(backbone, False)
        for block in backbone_last_block:
            set_requires_grad(block, True)
        set_requires_grad(classifier, True)
    elif phase == "full":
        set_requires_grad(backbone, True)
        set_requires_grad(classifier, True)
    else:
        raise ValueError("Unknown phase argument!")

    return backbone, classifier, checkpoint.get('epoch', None)

if __name__ == "__main__":

    args = get_argparser()
    args.classifier_type = 'arcface' # 'cosface' # 'arcface'  # Ensure classifier type is set for ArcFace
    args.phase = 'last_block' # 'last_block'  # Set phase for loading the model
    args.grayscale = False  # Set grayscale if needed
    args.model_name = 'mobilefacenet'  # Set model name
    args.optimizer = 'adamw'  # Set optimizer type

    test_loader_, meta_data = get_data_loader(
        root_dir=args.root_dir,
        train=True,
        crop_size=args.crop_size,
        batch_size=args.train_batch_size,
        grayscale=args.grayscale,
        aug_type=args.aug_type,
        shuffle=False,
    )
    """Trained including Challenging Faces"""
    snapshot_path = r"C:\Users\KU500996\Downloads\face_recog\mobilefacenet_arcface"
    
    backbone, _, _ = load_model_last_block(
        args,
        num_classes=meta_data['num_classes'],
        snapshot_path=snapshot_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    color = 'gray' if args.grayscale else 'rgb'
    embedding_path = snapshot_path + f"/{args.model_name}_{args.classifier_type}_{args.phase}_{color}_face_gallery.npz"
    if not os.path.exists(embedding_path):
        extract_embeddings(
            backbone=backbone,
            data_dir=args.root_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            crop_size=(112, 112),
            grayscale=args.grayscale,
            batch_size=32,
            out_path=embedding_path,
        )

    test_samples_dir = "./test_samples"
    # -- Detector
    detector = RetinaFacePyPIAdapter(threshold=0.5)
    
    # Loop through all images in the test_samples directory
    for image_file in os.listdir(test_samples_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(test_samples_dir, image_file)
            name = os.path.basename(image_path).split('.')[0]
            print(f"Processing image: {name}")
            
            # -- Call recognition
            recognize_unlabeled_faces_image(
                backbone=backbone,
                gallery_npz=embedding_path,    # Path to extracted gallery embeddings
                face_detector=detector,
                image_path=image_path,   # or image_array=...
                device='cuda' if torch.cuda.is_available() else 'cpu',
                crop_size=args.crop_size,
                grayscale=args.grayscale,
                topk=1,         # You can increase for more candidates
                threshold=0.1,  # Optionally set a similarity threshold for "Unknown"
                show=False,
                save_path=snapshot_path + f"/{name}_{args.model_name}_{args.classifier_type}_annotated_test.jpg"
            )

    test_video_dir = "./test_samples"
    for video_file in os.listdir(test_video_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(test_video_dir, video_file)
            video_name = os.path.basename(video_path).split('.')[0]
            print(f"Processing video: {video_name}")
    
            recognize_unlabeled_faces_video(
                backbone=backbone,
                gallery_npz=embedding_path,
                face_detector=detector,
                video_path=video_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                crop_size=(112,112),
                topk=1,
                threshold=0.4,
                show=False,
                save_path=snapshot_path + f"/{video_name}_{args.model_name}_{args.classifier_type}_annotated_video.mp4",
                search_gallery=search_gallery
            )


# python test_model.py --classifier_type combined

