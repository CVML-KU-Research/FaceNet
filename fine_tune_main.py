## %%

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from networks.smooth_ce_loss import SmoothCrossEntropyLoss
from networks.models_facenet import Backbone, MobileFaceNet, Arcface, CosFace

from data_loader import get_data_loader
from utils.utils import generate_snapshot_path
from utils.visualizer import plot_training_history, show_sample_images

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Set to your desired GPU ID

def get_argparser():
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
    parser.add_argument('--pretrained_backbone_path', type=str, default="mobile_weights/model_mobilefacenet.pth", help='Pretrained backbone path')
    parser.add_argument('--pretrained_head_path', type=str, default=None, help='Pretrained classifier/head path')

    parser.add_argument('--train_batch_size', type=int, default=16, help='Train batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
    parser.add_argument('--target_acc', type=float, default=95.0, help='Target accuracy for switching phase')

    parser.add_argument('--optimizer', type=str, default='adamw', choices=['SGD', 'adamw'], help='Optimizer')
    parser.add_argument('--max_epoch', type=int, default=500, help='Max epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=10, help='LR step')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='LR decay')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')

    parser.add_argument('--save_plot', action='store_true', help='Save training plots')
    parser.add_argument('--phase', type=str, default='head_only', choices=['head_only', 'last_block'], help='Training phase')
    parser.add_argument('--random_seed', type=int, default=2020, help='Random seed')

    return parser.parse_args()


# Seed everything to avoid non-determinism
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_requires_grad(module, requires_grad=True):
    for param in module.parameters():
        param.requires_grad = requires_grad

def set_bn_eval(module):
    """
    Set all BatchNorm layers in the module to eval() mode (i.e., freeze their running mean/var).
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()

def build_model(args, num_classes, phase="head_only"):
    """
    Build and configure the model + head for different training phases.
    Handles BatchNorm freezing and last block unfreezing for both MobileFaceNet and ResNet-style backbone.
    """
    # ---- Build backbone ----
    if args.model_name.lower() == "mobilefacenet":
        model = MobileFaceNet(embedding_size=args.embedding_size).cuda()
        backbone_last_block = [
            model.conv_5, model.conv_6_sep, model.conv_6_dw,
            model.conv_6_flatten, model.linear, model.bn
        ]
        # For BN freezing: all top-level blocks in MobileFaceNet
        mobilefacenet_blocks = [
            model.conv1, model.conv2_dw, model.conv_23, model.conv_3,
            model.conv_34, model.conv_4, model.conv_45,
            model.conv_5, model.conv_6_sep, model.conv_6_dw,
            model.conv_6_flatten, model.linear, model.bn
        ]
    else:
        model = Backbone(num_layers=50, drop_ratio=0.4, mode='ir_se').cuda()
        backbone_last_block = [model.body[-1]]

    # ---- Load pre-trained weights if specified ----
    if getattr(args, 'pretrained_backbone_path', None):
        print(f"Loading backbone weights from {args.pretrained_backbone_path}")
        model.load_state_dict(
            torch.load(args.pretrained_backbone_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'),
            strict=False
        )

    # ---- Build classifier head ----
    if args.classifier_type.lower() == "arcface":
        arcface_head = Arcface(embedding_size=args.embedding_size, classnum=num_classes, s=32.0, m=0.4).cuda()
        if getattr(args, 'pretrained_head_path', None):
            print(f"Loading arcface_head weights from {args.pretrained_head_path}")
            arcface_head.load_state_dict(
                torch.load(args.pretrained_head_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'),
                strict=False
            )
    elif args.classifier_type.lower() == "cosface":
        arcface_head = CosFace(embedding_size=args.embedding_size, classnum=num_classes, s=30.0, m=0.4).cuda()
        if getattr(args, 'pretrained_head_path', None):
            print(f"Loading cosface_head weights from {args.pretrained_head_path}")
            arcface_head.load_state_dict(
                torch.load(args.pretrained_head_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'),
                strict=False
            )
    # elif args.classifier_type.lower() == "combined":
    #     arcface_head = CombinedLossMargin(num_classes=num_classes, feat_dim=args.embedding_size, device='cuda', s=32, m1=0.2, m2=0.35)
    elif args.classifier_type.lower() == "fc":
        arcface_head = nn.Linear(args.embedding_size, num_classes).cuda()
    else:
        raise ValueError(f"Unknown classifier type: {args.classifier_type}")

    # ---- Set requires_grad and BatchNorm behavior ----
    if phase == "head_only":
        set_requires_grad(model, False)
        set_requires_grad(arcface_head, True)
        set_bn_eval(model)  # freeze all BNs
    elif phase == "last_block":
        set_requires_grad(model, False)
        set_requires_grad(arcface_head, True)
        if args.model_name.lower() == "mobilefacenet":
            # Unfreeze last block only, freeze others
            for block in backbone_last_block:
                set_requires_grad(block, True)
            # Freeze BN in all frozen blocks
            for b in mobilefacenet_blocks:
                if b not in backbone_last_block:
                    set_bn_eval(b)
        else:  # ResNet-style backbone
            for block in backbone_last_block:
                set_requires_grad(block, True)
            # Set BN eval for all frozen blocks except the last one
            for i, block in enumerate(model.body):
                if i != len(model.body) - 1:
                    set_bn_eval(block)
    elif phase == "full":
        set_requires_grad(model, True)
        set_requires_grad(arcface_head, True)
    else:
        raise ValueError("Unknown phase argument!")

    return model, arcface_head


## %%
def build_optimizer(args, model, arcface_head):
    # Only parameters with requires_grad=True will be updated
    params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
             list(filter(lambda p: p.requires_grad, arcface_head.parameters()))
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    return optimizer

def train_model(args, model, arcface_head, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, snapshot_path, patience=40):
    """
    Training loop for face recognition models, supporting best model snapshot saving.
    """
    scaler = GradScaler()
    best_acc = 0.0
    best_epoch = -1
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            with autocast():
                embeddings = model(inputs)
                if args.classifier_type.lower() == "arcface":
                    outputs = arcface_head(embeddings, labels)
                elif args.classifier_type.lower() == "cosface":
                    outputs = arcface_head(embeddings, labels)
                elif args.classifier_type.lower() == "combined":
                    outputs = arcface_head(embeddings, labels)
                else:
                    # For FC or other heads, no labels needed
                    outputs = arcface_head(embeddings)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, list(model.parameters()) + list(arcface_head.parameters())), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)
        # --- Validation phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                embeddings = model(inputs)
                if args.classifier_type.lower() == "arcface":
                    outputs = arcface_head(embeddings, labels)
                elif args.classifier_type.lower() == "cosface":
                    outputs = arcface_head(embeddings, labels)
                elif args.classifier_type.lower() == "combined":
                    outputs = arcface_head(embeddings, labels)
                else:
                    # For FC or other heads, no labels needed
                    outputs = arcface_head(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        val_loss /= len(val_loader)

        if epoch >= 5:
            scheduler.step(val_loss)

        # --- Save best model and record best epoch ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            early_stop_count = 0
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'arcface_state_dict': arcface_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(snapshot_path, f'best_model_epoch{best_epoch}_acc{best_acc:.2f}.pth'))
        else:
            early_stop_count += 1

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%')

        if early_stop_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    # --- Save summary log ---
    summary_path = os.path.join(snapshot_path, "train_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Best Val Acc: {best_acc:.4f} at epoch {best_epoch}\n")
        f.write(f"Args: {vars(args)}\n")
        f.write(f"Training history: {history}\n")

    return history, best_acc, best_epoch

def main():
    """
    Main training pipeline for face recognition models.
    Dynamically builds loaders, meta info, and snapshot paths for all phases.
    """
    args = get_argparser()
    seed_everything(args.random_seed)

    # ---- 1. Get Data Loaders and Meta ----
    train_loader, train_meta = get_data_loader(
        root_dir=args.root_dir,
        train=True,
        crop_size=args.crop_size,
        batch_size=args.train_batch_size,
        grayscale=args.grayscale,
        aug_type=args.aug_type,
    )
    test_loader, _ = get_data_loader(
        root_dir=args.root_dir,
        train=False,
        crop_size=args.crop_size,
        batch_size=args.test_batch_size,
        grayscale=args.grayscale,
        aug_type=args.aug_type,
        shuffle=False
    )
    print(f"Total train: {len(train_loader.dataset)}, test: {len(test_loader.dataset)}")
    # show_sample_images(train_loader, test_loader, idx_train=0, idx_test=0)

    # ---- 2. Training Loop by Phase ----
    print("Training Face Recognition Models")
    num_classes = train_meta["num_classes"]
    # for phase in ['head_only', 'last_block']:
    for phase in ['last_block']:
        print(f"\nPhase: {phase.upper()}")

        # Generate snapshot_path including timestamp, phase, etc.
        snapshot_path = generate_snapshot_path(
            base_dir=args.checkpoints_path,
            meta=train_meta,
            args=args,
            phase=phase,
            add_timestamp=True
        )
        print(f"Snapshot path for this phase: {snapshot_path}")

        # Build model/head
        model, arcface_head = build_model(args, num_classes, phase=phase)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                        sum(p.numel() for p in arcface_head.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable}")

        # Loss and optimizer
        criterion = (
            SmoothCrossEntropyLoss().cuda()
            if getattr(args, 'loss_type', 'ce') == "smooth_ce"
            else nn.CrossEntropyLoss().cuda()
        )
        optimizer = build_optimizer(args, model, arcface_head)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6, verbose=True
        )

        # Train
        history, best_acc, best_epoch = train_model(
            args=args,
            model=model,
            arcface_head=arcface_head,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=args.max_epoch if phase == "head_only" else args.max_epoch // 2,
            snapshot_path=snapshot_path,
            patience=30
        )
        print(f"Best val acc after {phase}: {best_acc:.4f} at epoch {best_epoch}")
        if args.save_plot:
            print(f"Saving training history plot to {snapshot_path}")
            if history: # Ensure history is not empty
                plot_training_history(history, snapshot_path, args)

        # Stop if target acc reached
        if phase == "head_only" and best_acc >= args.target_acc:
            print("Target accuracy reached. Stopping further fine-tuning.")
            break

if __name__ == "__main__":
    main()

## %%
# python fine_tune_main.py --save_plot --model_name mobilefacenet --classifier_type arcface --phase last_block --optimizer adamw
# python fine_tune_main.py --save_plot --model_name mobilefacenet --classifier_type cosface --phase last_block --optimizer adamw
# python fine_tune_main.py --save_plot --model_name mobilefacenet --classifier_type FC --phase last_block --optimizer adamw
# python fine_tune_main.py --save_plot --model_name ir_se --classifier_type arcface --phase last_block --optimizer adamw
