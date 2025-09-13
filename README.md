# AOCloss: Adaptive Centroid Shift Loss for Audio Deepfake Detection

## 📖 Overview
This repository provides an implementation of the **Adaptive Centroid Shift Loss (AOCloss)** method for **Audio Deepfake Detection**, as described in the corresponding [research paper](https://www.isca-archive.org/interspeech_2024/kim24b_interspeech.pdf). The approach employs a one-class learning framework that continuously adapts a centroid to represent bonafide audio embeddings while maximizing the distance of spoof embeddings.

## 🛠️ Usage
### 1. **Initialization**
```python
from AOC_loss import AOCloss

# Initialize with desired embedding dimension
criterion = AOCloss(embedding_dim=512)
```

### 2. **Using the AOC loss**
```python
loss = criterion(embeddings, labels)
```
- `embeddings`: Tensor of shape `(batch_size, embedding_dim)`.
- `labels`: Binary tensor where `0` represents bonafide samples and `1` represents spoof samples.

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

## 📄 Citations
If you use this repository in your work, please cite it as follows:
```
@misc{Weizman2025_OneClassACS_GitHub,
  author       = {Weizman, Avishai},
  title        = {One-Class Learning with Adaptive Centroid Shift for Audio Deepfake Detection — Implementation (GitHub)},
  howpublished = {GitHub repository},
  year         = {2025},
  url          = {https://github.com/avishai111/One-Class-Learning-with-Adaptive-Centroid-Shift-for-Audio-Deepfake-Detection},
  note         = {Related to the paper “One-Class Learning with Adaptive Centroid Shift for Audio Deepfake Detection” by Kim, Jang, and Kim, Interspeech 2024.}
}
```
This github implementation based on the following paper:
```
@inproceedings{kim24b_interspeech,
  title     = {One-class learning with adaptive centroid shift for audio deepfake detection},
  author    = {Hyun Myung Kim and Kangwook Jang and Hoirin Kim},
  year      = {2024},
  booktitle = {Interspeech 2024},
  pages     = {4853--4857},
  doi       = {10.21437/Interspeech.2024-177},
  issn      = {2958-1796},
}
```

## 🙋‍♂️ Support
For issues or questions, feel free to open an issue in the repository.

## 📢 Acknowledgments
- This implementation is inspired by [research paper](https://www.isca-archive.org/interspeech_2024/kim24b_interspeech.pdf).

- 
## 📬 Contact

If you have questions, feedback, or want to collaborate, feel free to reach out:

 📧 Email: [Avishai Weizman](mailto:wavishay@post.bgu.ac.il)  

 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)

 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)  
 
 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
 
