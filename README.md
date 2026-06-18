# 🔬 Mikroplastik Tespiti — YOLOv8 / YOLO11

Mikroskobik su görüntülerinde mikroplastikleri tespit eden ve türlerine göre sınıflandıran derin öğrenme projesi.

---

## 📌 Proje Hakkında

Bu proje, su örneklerinin mikroskop görüntülerinden mikroplastikleri otomatik olarak tespit etmek ve türlerine göre sınıflandırmak amacıyla geliştirilmiştir. **fiber, film, fragment ve pellet** olmak üzere 4 farklı mikroplastik türü tanınmaktadır. Model olarak YOLO mimarisi kullanılmış; farklı model varyantları karşılaştırmalı olarak değerlendirilmiştir.

---

## 🧠 Kullanılan Modeller

| Model     | Notlar                          |
|-----------|---------------------------------|
| YOLOv8s   | Temel model (en iyi sonuç)      |
| YOLOv8m   | Karşılaştırma amaçlı            |
| YOLO11s   | Karşılaştırma amaçlı            |
| YOLO11m   | Karşılaştırma amaçlı            |

> Farklı model varyantları arasında performans farkı belirgin değildi; temel model olarak **YOLOv8s** kullanılmıştır.

---

## 🏷️ Sınıflar

Modelin tespit ettiği mikroplastik türleri:

| Sınıf        | Açıklama                         |
|--------------|----------------------------------|
| **fiber**    | İplik / lif şeklinde parçacıklar |
| **film**     | İnce film tabakası şeklinde      |
| **fragment** | Düzensiz köşeli parçacıklar      |
| **pellet**   | Yuvarlak / granül formlar        |

---

## 📊 Veri Seti

Eğitim verisi **Roboflow** platformundan temin edilmiştir. Görüntüler mikroskobik su örneklerine aittir.

| Küme       | Görüntü Sayısı |
|------------|----------------|
| Train      | 8.394          |
| Validation | 1.348          |
| Test       | 888            |
| **Toplam** | **10.630**     |

---

## ⚙️ Eğitim Parametreleri

| Parametre      | Değer    |
|----------------|----------|
| Model          | YOLOv8s  |
| Image size     | 1280     |
| Batch size     | 16       |
| Epochs         | 50       |
| Patience       | 15       |
| Optimizer      | AdamW    |
| Learning rate  | 0.001    |
| Cosine LR      | True     |
| Dropout        | 0.1      |
| Platform       | Lightning AI (NVIDIA L4) |

**Augmentation:** mosaic, mixup, copy_paste, degrees, translate, scale, shear, perspective, flip, HSV

---

## 📈 Sonuçlar

YOLOv8s modeli ile elde edilen performans metrikleri:

| Metrik      | Değer  |
|-------------|--------|
| mAP@50      | 0.8226 |
| mAP@50-95   | 0.4408 |

---

## 🚀 Kurulum ve Kullanım (Lightning AI)

Bu proje **[Lightning AI](https://lightning.ai)** üzerinde geliştirilmiştir. Aşağıdaki adımları takip ederek aynı ortamda çalıştırabilirsiniz.

### 1. Lightning AI Studio Oluştur

[lightning.ai](https://lightning.ai) adresine gidip yeni bir **Studio** aç. GPU olarak **L4** veya üstünü seç.

### 2. Veri Setini Hazırla

Roboflow'dan çeşitli datasetleri birleştirdiğimiz birleşik datasetimiz büyük olduğundan dolayı githuba atamadık ama aşağıdaki linkten ulaşıp yukarıda belirtilen uygun dosya yapısındaki yerine taşıyın lütfen.

Dataset Linki: https://drive.google.com/file/d/1AmL5-iPQ1Wx9g4TGHMyZs1RcBHtpvI1R/view?usp=sharing

```python
import zipfile, os
from pathlib import Path

ZIP_PATH    = 'merged_dataset.zip'
DATASET_DIR = 'dataset'

os.makedirs(DATASET_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(DATASET_DIR)

# Klasör yapısını doğrula
for split in ['train', 'valid', 'test']:
    img_path   = Path(f'{DATASET_DIR}/merged_dataset/{split}/images')
    label_path = Path(f'{DATASET_DIR}/merged_dataset/{split}/labels')
    if img_path.exists():
        print(f'{split}: {len(list(img_path.glob("*")))} görüntü')
```

### 3. data.yaml Dosyasını Güncelle

Lightning AI'da Google Drive olmadığı için `data.yaml` içindeki yolların mutlak (absolute) path olması gerekir:

```python
import yaml, os

DATASET_DIR = 'dataset'
yaml_path   = os.path.abspath(f'{DATASET_DIR}/merged_dataset/data.yaml')

with open(yaml_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['train'] = os.path.abspath(f'{DATASET_DIR}/merged_dataset/train/images')
cfg['val']   = os.path.abspath(f'{DATASET_DIR}/merged_dataset/valid/images')
cfg['test']  = os.path.abspath(f'{DATASET_DIR}/merged_dataset/test/images')

if 'path' in cfg:
    del cfg['path']

with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
```

### 4. Modeli Eğit

```python
from ultralytics import YOLO
import os

model       = YOLO('yolov8s.pt')
project_dir = os.path.abspath('mikroplastik_egitimleri')

results = model.train(
    data         = yaml_path,
    imgsz        = 1280,
    batch        = 16,
    workers      = 4,
    epochs       = 50,
    patience     = 15,
    optimizer    = 'AdamW',
    lr0          = 0.001,
    cos_lr       = True,
    dropout      = 0.1,
    mosaic       = 1.0,
    mixup        = 0.15,
    copy_paste   = 0.1,
    project      = project_dir,
    name         = 'yolov8s_adamw',
    save         = True,
    save_period  = 10,
    plots        = True,
    device       = 0
)
```

Eğitim tamamlandığında ağırlıklar `mikroplastik_egitimleri/yolov8s_adamw/weights/best.pt` konumuna kaydedilir.

### 5. Tahmin Yap

```python
from ultralytics import YOLO

model   = YOLO('best.pt')
results = model.predict(
    source  = 'dataset/merged_dataset/test/images',
    conf    = 0.25,
    imgsz   = 1280,
    save    = True,
    project = 'test_sonuclari',
    name    = 'tahminler'
)
```

Sonuçlar `test_sonuclari/tahminler/` klasörüne kaydedilir.

---

## 📁 Dosya Yapısı

```
Microplastic-Detection-ANN/
├── Untitled.ipynb       # Eğitim ve değerlendirme notebook'u
├── .gitignore
├── merged_dataset.zip
├── LICENSE
└── README.md
```

---

## 👥 Ekip

|Öğrenci No |İsim              | GitHub                         |
|-------    |------            |--------                        |
|032490047  |Büşra DERELİ      |https://github.com/Busra-Dereli |


---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
