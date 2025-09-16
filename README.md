# Face Recognition Fine‑Tuning — Quick Start

1) Setup
Python 3.8+ and PyTorch (CUDA optional but recommended).
Install deps (minimal): numpy, tqdm, torch, torchvision.

2) Data layout

Place images under class-labeled folders:

```<data_root>/
  <class_id_1>/ 
    img1.jpg
    img2.png
    ...
  <class_id_2>/ 
    img3.jpg
    img4.png
    ...
  <class_id_3>/ 
    ...
```

3) Start fine-tuning with four face images

- `--model_name`: `mobilefacenet` | `ir_se`  
- `--classifier_type`: `arcface` | `cosface` | `FC`  
- `--aug_type`: `standard` | `strong` | `none`  
- `--phase`: `head_only` | `last_block`  
- `--train_batch_size` / `--test_batch_size`  
- `--max_epoch`, `--optimizer (adamw | SGD)`  
- `--save_plot` → save loss/acc plots

MobileFaceNet + ArcFace (last block fine‑tune):
  
```python .\fine_tune_main.py --root_dir .\data --model_name mobilefacenet --classifier_type arcface --phase last_block --optimizer adamw --save_plot```

MobileFaceNet + CosFace
  
```python .\fine_tune_main.py --root_dir .\data --model_name mobilefacenet --classifier_type cosface --phase last_block --optimizer adamw --save_plot```

4) Testing
```python test_model.py --classifier_type combined ```

