# ACSloss: Adaptive Centroid Shift Loss for Audio Deepfake Detection

## 📖 Overview
This repository provides an implementation of the **Adaptive Centroid Shift Loss (ACSloss)** method for **Audio Deepfake Detection**, as described in the corresponding research paper. The approach employs a one-class learning framework that continuously adapts a centroid to represent bonafide audio embeddings while maximizing the distance of fake embeddings.

## ⚙️ Features
- **Dynamic Centroid Update**: Continuously updates the centroid based on bonafide embeddings.
- **Cosine Similarity Loss**: Ensures that bonafide samples are closer to the centroid and fake samples are pushed further away.
- **Robust One-Class Learning**: Optimized for scenarios where fake data is highly variable.
- **PyTorch-Based**: Easy integration with PyTorch workflows.

## 🛠️ Usage
### 1. **Initialization**
```python
from loss import ACSloss

# Initialize with desired embedding dimension
criterion = ACSloss(embedding_dim=512)
```

### 2. **Using the ACS loss**
```python
loss = criterion(embeddings, labels)
```
- `embeddings`: Tensor of shape `(batch_size, embedding_dim)`.
- `labels`: Binary tensor where `0` represents bonafide samples and `1` represents fake samples.

### 3. **Centroid Update**
The centroid is automatically updated during the forward pass.

```python
criterion.update_centroid(bonafide_embeddings)
```

## ✅ Requirements
- PyTorch >= 1.10
- Python >= 3.8

## ❓ Troubleshooting
- **`ValueError: Centroid has not been initialized`**:
  - Ensure the batch contains bonafide samples.

- **Negative Loss**:
  - This is expected due to the range of cosine similarity between `-1` and `1`.

## 📄 Citation
If you use this implementation in your research, please consider citing the following paper:

```
@inproceedings{acs_audio_deepfake,
  title={One-Class Learning with Adaptive Centroid Shift for Audio Deepfake Detection},
  author={Authors' Names},
  booktitle={Conference Name},
  year={Year},
  publisher={Publisher}
}
```

## 🙋‍♂️ Support
For issues or questions, feel free to open an issue in the repository.

## 📢 Acknowledgments
- This implementation is inspired by the research on Adaptive Centroid Shifting for one-class learning in audio deepfake detection.
