# steerable-medical-dino

## 1. Setup

```bash
pip install git+https://github.com/larosi/steerable-medical-dino.git
```

## 2. How to use it
```python
from medical_sdino import load_sdino, get_transforms

transforms = get_transforms(img_size=448)
model_path = r'models\model.pth'
model = load_sdino(model_path)

with torch.no_grad():
    images = transforms(images)

    features = model.forward_features(images)
    patch_tokens = features['x_norm_patchtokens'] # tensor of shape [batch_size, number_of_patches, 768]
    cls_token = features["x_norm_clstoken"] # tensor of shape [batch_size, 768])
```