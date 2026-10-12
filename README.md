# Tugas 05 — Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

Brian Nicholas Tedjo — 2802403183
Deep Learning, Sesi 05: *Advanced CNNs and Transfer Learning*

Studi komparatif pelatihan ResNet-18 **from scratch**, **feature extraction**, dan **progressive fine-tuning** pada subset 10 kelas Oxford-IIIT Pet (~2.000 citra), termasuk uji kontrol data augmentation dan perbandingan arsitektur efisien (EfficientNet-B0, MobileNetV3-Small).

## Struktur

| Path | Isi |
|---|---|
| `answer.ipynb` | Notebook utama (sudah dieksekusi): data pipeline, eksperimen B1–B5, tabel & kurva (C1), failure analysis (C3), Grad-CAM (C5) |
| `report.md` | Laporan analisis (Bagian A & C) |
| `logs/history.csv` | Metrik per epoch semua kondisi (train/val loss & accuracy, waktu) |
| `logs/summary.csv`, `logs/metrics.json` | Ringkasan per kondisi + konfigurasi eksperimen & versi library |
| `logs/misclassified_best.csv` | Sampel test yang salah diprediksi model terbaik |
| `figures/` | Kurva loss B1–B3, kurva akurasi B4, confusion matrix, sampel misklasifikasi, Grad-CAM |

## Dataset

Oxford-IIIT Pet via `torchvision.datasets.OxfordIIITPet` (diunduh otomatis ke `./oxford-iiit-pet/`).
Subset 10 kelas — kucing: Abyssinian, Bengal, Birman, Persian, Siamese; anjing: Beagle, Boxer, Chihuahua, Pug, Samoyed.
Split resmi `trainval` + `test` digabung lalu dibagi ulang secara stratified 70/15/15 (seed 42): **1.397 train / 299 val / 300 test**.

## Protokol eksperimen

ResNet-18 (`IMAGENET1K_V1`), input 224×224 dengan normalisasi ImageNet, batch size 32, `AdamW` (weight decay 1e-2), `CrossEntropyLoss`, early stopping pada validation loss (patience 5, maks. 50 epoch), seed 42 untuk Python/NumPy/PyTorch (`cudnn.deterministic=True`). Augmentasi (RandomResizedCrop, HorizontalFlip, Rotation 15°, ColorJitter) hanya pada train set.

| Kode | Kondisi |
|---|---|
| B1 | From scratch (`weights=None`), lr 1e-3 |
| B2 | Feature extraction: backbone beku (termasuk statistik BatchNorm), hanya `fc` (5.130 params), lr 1e-3 |
| B3 | Fine-tuning `layer4` (lr 1e-4) + `fc` (lr 1e-3) |
| B4 | B3 tanpa augmentasi vs. B3 dengan augmentasi |
| B5 | Konfigurasi terbaik (dipilih via validasi) pada EfficientNet-B0 & MobileNetV3-Small |

## Hasil ringkas

Sumber: `logs/summary.csv`. Dijalankan pada NVIDIA GeForce RTX 3080, torch 2.14.0+cu132.

| Kondisi | Trainable params | Epoch terbaik | Waktu latih (s) | Val acc | Test Macro F1 |
|---|---|---|---|---|---|
| B1 scratch | 11,181,642 | 15 | 440.9 | 0.5251 | 0.5199 |
| B2 feature extraction | 5,130 | 25 | 665.0 | 0.9465 | 0.9534 |
| B3 fine-tuning (aug) | 8,398,858 | 7 | 268.5 | 0.9565 | 0.9567 |
| B4 fine-tuning (no aug) | 8,398,858 | 10 | 323.2 | 0.9699 | 0.9634 |
| B5 EfficientNet-B0 | 1,142,202 | 7 | 261.4 | 0.9866 | 0.9766 |
| B5 MobileNetV3-Small | 951,642 | 3 | 161.9 | 0.9465 | 0.9701 |

## Cara menjalankan

```bash
pip install -r requirements.txt
jupyter notebook answer.ipynb   # Run All; dataset & bobot pretrained diunduh otomatis
```

Atau unggah `answer.ipynb` ke Google Colab (runtime GPU T4) dan jalankan seluruh sel. Checkpoint model disimpan ke `checkpoints/` (tidak disertakan di repo karena ukurannya).

## Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
- Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and Dogs. *CVPR*.
- PyTorch & torchvision documentation — https://pytorch.org/docs
