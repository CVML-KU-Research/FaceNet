
import cv2
import torch
import numpy as np
from torchviz import make_dot
from IPython.display import display, Image

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

from torchvision import transforms


class Visualizer(object):

    def __init__(self):
        self.iters = {}
        self.lines = {}

    def display_current_results(self, iters, x, name='train_loss'):
        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        plt.figure(name)
        plt.clf()
        plt.plot(self.iters[name], self.lines[name], label=name)
        plt.xlabel('Iterations')
        plt.ylabel(name)
        plt.title(name)
        plt.legend()
        plt.pause(0.001)

    def display_roc(self, y_true, y_pred):
        fpr, tpr, ths = roc_curve(y_true, y_pred)
        plt.figure('roc')
        plt.clf()
        plt.plot(fpr, tpr, label='roc')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.pause(0.001)

        
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axs[0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axs[0].plot(epochs, val_losses, label='Val Loss', marker='o')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy plot
    axs[1].plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
    axs[1].plot(epochs, val_accuracies, label='Val Accuracy', marker='o')
    axs[1].set_title('Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def recognize_faces_in_video(
    model,
    class_names,
    face_detector,
    video_path,
    device='cuda',
    show=False,          # Show frames live (True/False)
    save_path=None,      # Path to save output video, e.g. 'output.mp4'
    max_frames=None      # Max number of frames to process (None for all)
):
    """
    Recognize faces in a video using a trained model.
    Args:
        model: Trained PyTorch model (outputs logits).
        class_names: List of class (person) names.
        face_detector: Object with .detect_faces(frame) -> [(x1,y1,x2,y2), ...].
        video_path: Path to input video.
        device: 'cuda' or 'cpu'.
        show: If True, display frames live (matplotlib if in notebook, cv2.imshow if script).
        save_path: If given, save output video to this file.
        max_frames: Stop after this many frames (for demo/testing).
    """
    import sys
    from matplotlib import pyplot as plt

    model.eval()
    model.to(device)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.416, 0.367, 0.322], std=[0.191, 0.195, 0.202])  # Replace with your stats
    ])

    cap = cv2.VideoCapture(video_path)
    out = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        orig_frame = frame.copy()
        face_bboxes = face_detector.detect_faces(frame)

        for bbox in face_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            face_img = frame[y1:y2, x1:x2]
            if face_img.shape[0] == 0 or face_img.shape[1] == 0:
                continue
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(face_tensor)
                pred = logits.argmax(dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0, pred].item()
            label = f"{class_names[pred]} ({confidence:.2f})"
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(orig_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # ----- Handle video writing -----
        if save_path is not None:
            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = orig_frame.shape[:2]
                out = cv2.VideoWriter(save_path, fourcc, 20, (w, h))
            out.write(orig_frame)

        # ----- Handle video showing -----
        if show:
            # If in notebook, use matplotlib, else use cv2.imshow
            if 'ipykernel' in sys.modules:
                # Jupyter: matplotlib
                from matplotlib import pyplot as plt
                plt.imshow(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            else:
                # Script: OpenCV window
                cv2.imshow('Face Recognition', orig_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
    if show and not 'ipykernel' in sys.modules:
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")


def recognize_faces_in_image(
    model,
    class_names,
    face_detector,
    image_path=None,      # Path to the input image
    image_array=None,     # Alternatively, provide the loaded image as a NumPy array
    device='cuda',
    show=True,            # Show image with matplotlib
    save_path=None        # Save annotated image if desired
):
    """
    Recognize and annotate faces in a single image.

    Args:
        model: Trained PyTorch model.
        class_names: List of class names corresponding to model outputs.
        face_detector: Object with detect_faces(img) -> [(x1, y1, x2, y2), ...]
        image_path: Path to the input image (if not providing image_array).
        image_array: Input image as a NumPy array (BGR).
        device: 'cuda' or 'cpu'.
        show: Whether to display the result (matplotlib).
        save_path: If provided, save the annotated image to this path.
    """
    # Load the image
    if image_array is not None:
        img = image_array.copy()
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

    orig_img = img.copy()

    # Detect faces
    face_bboxes = face_detector.detect_faces(img)
    print(f"Detected {len(face_bboxes)} faces.")

    # Preprocessing (same as during training!)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.416, 0.367, 0.322], std=[0.191, 0.195, 0.202])
    ])

    model.eval()
    model.to(device)

    for bbox in face_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        face_img = img[y1:y2, x1:x2]
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue
        face_tensor = preprocess(face_img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(face_tensor)
            pred = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred].item()
        label = f"{class_names[pred]} ({confidence:.2f})"
        # Draw bounding box and label
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(orig_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Show the annotated image
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Optionally save the output
    if save_path is not None:
        cv2.imwrite(save_path, orig_img)
        print(f"Annotated image saved to {save_path}")

    return orig_img, len(face_bboxes)

# Visualization function for multiple models
def plot_training_progress(results):
    num_models = len(results)
    plt.figure(figsize=(15, num_models * 5))
    
    for i, (model_name, result) in enumerate(results.items(), 1):
        train_losses = result['history']['train_loss']
        val_losses = result['history']['val_loss']
        train_accs = result['history']['train_acc']
        val_accs = result['history']['val_acc']
        
        # Loss Plot
        plt.subplot(num_models, 2, 2 * i - 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_name} - Loss')

        # Accuracy Plot
        plt.subplot(num_models, 2, 2 * i)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'{model_name} - Accuracy')
    
    plt.tight_layout()
    plt.show()

# Visualization function for multiple models
def plot_training_progress_(results):
    num_models = len(results)
    plt.figure(figsize=(15, num_models * 5))
    
    for i, (model_name, result) in enumerate(results.items(), 1):
        train_losses = result['train_loss']
        val_losses = result['val_loss']
        train_accs = result['train_acc']
        val_accs = result['val_acc']

        # Loss Plot
        plt.subplot(num_models, 2, 2 * i - 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{model_name} - Loss')

        # Accuracy Plot
        plt.subplot(num_models, 2, 2 * i)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title(f'{model_name} - Accuracy')
    
    plt.tight_layout()
    plt.show()

def visualize_model_architecture(model, input_size=(3, 112, 112), batch_size=4):
    model.eval()  # Set the model to evaluation mode
    x = torch.randn(batch_size, *input_size).cuda()
    try:
        y = model(x)
        dot = make_dot(y, params=dict(model.named_parameters()))
        dot.render(f"{model.__class__.__name__}_model_architecture", format="png")
        display(Image(f"{model.__class__.__name__}_model_architecture.png"))
    except Exception as e:
        print(f"Error visualizing model architecture: {str(e)}")
        print("Skipping model architecture visualization.")


def plot_training_history(history, snapshot, args):
    plt.figure(figsize=(12, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(f"{snapshot}/{args.model_name}_opt_{args.optimizer}_training_history_{args.max_epoch}_epochs.png", dpi=300)
    plt.close()

def show_sample_images(train_loader, test_loader, idx_train=0, idx_test=0):
    """
    Display sample images from train and test loaders.

    Args:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        idx_train: Index of the sample to display from the training set.
        idx_test: Index of the sample to display from the testing set.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show train sample
    sample = train_loader.dataset[idx_train]
    img = sample[0]
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    ax[0].imshow(img)
    ax[0].set_title(f"Train-Label: {sample[1]}")
    ax[0].axis('off')

    # Show test sample
    sample = test_loader.dataset[idx_test]
    img = sample[0]
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    if img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    ax[1].imshow(img)
    ax[1].set_title(f"Test-Label: {sample[1]}")
    ax[1].axis('off')

    # plt.show()
    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{current_datetime}_sample_images_train_test_{idx_train}_{idx_test}.png", dpi=300)
    plt.close()


def arcface_recognize_faces_in_image(
    args,  # argparse or config object with model_name, classifier_type, etc.
    model,              # (backbone, classifier) tuple
    class_names,        # List of class names (ordered as in training)
    face_detector,      # .detect_faces(img) -> [(x1, y1, x2, y2), ...]
    image_path=None,    # Path to image (if image_array not provided)
    image_array=None,   # Optional: image as numpy array (BGR)
    device='cuda',
    show=True,
    save_path=None,
    grayscale=False,    # Pass True if your model expects grayscale (per config)
    crop_size=(112, 112), # Resize size; must match training
    mean=[0.3028, 0.2526, 0.2140],  # Use mean/std from your dataset stats/config
    std=[0.1909, 0.1914, 0.1904]
):
    """
    Recognize and annotate faces in a single image using a backbone-classifier pipeline.

    Args:
        model: (backbone, classifier) tuple. Each should be nn.Module and on correct device.
        class_names: List of class names corresponding to output indices.
        face_detector: Object with detect_faces(img) -> [(x1, y1, x2, y2), ...]
        image_path: Path to the input image (if image_array not provided).
        image_array: Input image as a NumPy array (BGR). If None, loads from image_path.
        device: 'cuda' or 'cpu'.
        show: If True, display annotated image with matplotlib.
        save_path: If given, save annotated image.
        grayscale: If True, converts detected face crops to grayscale before model input.
        crop_size: Face region will be resized to this before inference.
        mean, std: List. Should match training stats exactly.
    Returns:
        The annotated image (numpy array, BGR).
    """
    # 1. Load the image
    if image_array is not None:
        img = image_array.copy()
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
    orig_img = img.copy()

    # 2. Detect faces
    face_bboxes = face_detector.detect_faces(img)

    # 3. Build preprocessing transform (should match training)
    tr = []
    if grayscale:
        tr.append(transforms.Grayscale(num_output_channels=3))
    tr.extend([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    preprocess = transforms.Compose(tr)

    # 4. Set model to eval mode & move to device
    backbone, classifier = model
    backbone.eval()
    classifier.eval()
    backbone.to(device)
    classifier.to(device)

    # 5. Loop over faces
    for bbox in face_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        face_img = img[y1:y2, x1:x2]
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue
        face_pil = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = transforms.ToPILImage()(face_pil)
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = backbone(face_tensor)
            if args.classifier_type.lower() == 'arcface':
                logits = classifier(embedding, )
            pred = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred].item()
        label = f"{class_names[pred]} ({confidence:.2f})"
        # Draw bounding box and label
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(orig_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # 6. Show or save
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    if save_path is not None:
        cv2.imwrite(save_path, orig_img)
        print(f"Annotated image saved to {save_path}")

    return orig_img


def debug_batch_and_show(test_loader, meta_data, save_path="strong_aug_sample_for_embedding_norm.png", cols=4):
    """
    Debug a batch of images and display them in a grid.

    Args:
        test_loader: DataLoader for the test dataset.
        meta_data: Dictionary containing metadata, including 'class_names'.
        save_path: Path to save the output grid image.
        cols: Number of columns in the grid.
    """
    # Step 4: Debug batch and show
    imgs, labels = next(iter(test_loader))
    print(f"Batch: {imgs.shape}, Labels: {labels}")
    class_names = meta_data['class_names']
    print(f"Class names: {class_names}")

    rows = (imgs.shape[0] + cols - 1) // cols  # Calculate rows needed
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()  # Flatten the axes array for easy indexing

    for i in range(imgs.shape[0]):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = img_np.clip(0, 1)
        ax[i].imshow(img_np)
        ax[i].set_title(f"Label ID: {class_names[labels[i]]}")
        ax[i].axis("off")

    # Hide any unused subplots
    for i in range(imgs.shape[0], len(ax)):
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Grid image saved to {save_path}")