# Sentetik YOLO Veri Üretici

Drone veya sabit kamera görüntülerinin üzerine şeffaf arka planlı nesneler yerleştirerek YOLO nesne tespiti formatında sentetik veri seti üretir. Nesneler her görüntüde rastgele ölçeklenir, döndürülür, konumlandırılır; parlaklık ve bulanıklık dönüşümleri uygulanır. Görüntü ile eşleşen sınırlayıcı kutu etiketi otomatik oluşturulur.

> Bu proje tek nesne sınıfı üretir. Varsayılan sınıf `mannequin`'dir; farklı bir sınıf için `--class-name` kullanın. Tek sınıflı YOLO veri setinde `--class-id` değeri `0` olmalıdır.

## Özellikler

- JPG, JPEG, PNG, WebP ve BMP arka plan desteği
- Alfa kanallı nesnelerden görünür piksele göre YOLO kutusu üretimi
- Ölçek, dönüş, parlaklık, bulanıklık ve görünürlük ayarları
- Aynı `--seed` ve aynı girdilerle tekrarlanabilir üretim
- Mevcut çıktıları yanlışlıkla ezmeye karşı koruma
- Yeni üretimi `--start-index` ile mevcut veri setinin sonuna ekleme

## Gereksinimler

- Python 3.10 veya üzeri
- Pillow 10.0 veya üzeri

## Kurulum

Depoyu klonladıktan sonra sanal ortam oluşturun ve bağımlılığı yükleyin:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS için aktivasyon komutu:

```bash
source .venv/bin/activate
```

## Girdi verisini hazırlama

Klasör yapısı şöyle olmalıdır:

```text
data/
├── backgrounds/   # Arka plan fotoğrafları
└── mannequins/    # Şeffaf arka planlı nesne görselleri
```

İyi sonuç için:

1. `data/backgrounds/` içine modelin gerçek kullanım ortamına benzeyen, mümkünse farklı yükseklik, açı, ışık ve hava koşullarından arka planlar koyun.
2. `data/mannequins/` içine nesnenin sıkı kırpılmış, gerçekten şeffaf alfa kanalına sahip PNG/WebP görsellerini koyun. Beyaz veya kare arka planlı görseller uygun değildir.
3. Kaynak görsellerin kullanım ve dağıtım lisanslarını kontrol edin. Bu klasörler `.gitignore` içindedir; böylece büyük veya lisansı belirsiz dosyalar yanlışlıkla GitHub'a gönderilmez.
4. Eğitimde yalnızca sentetik veri kullanmayın. Sentetik veriyi gerçek görüntülerle karıştırın ve doğrulama/test kümelerini mümkünse yalnızca gerçek, birbirinden bağımsız görüntülerden oluşturun.

## Hızlı başlangıç

100 örnek üretmek için:

```powershell
python generate_synthetic.py --count 100 --seed 42
```

Sonuç varsayılan olarak şu yapıda oluşur:

```text
output/
├── classes.txt
├── images/
│   ├── synthetic_000000.jpg
│   └── ...
└── labels/
    ├── synthetic_000000.txt
    └── ...
```

Her etiket satırı YOLO biçimindedir:

```text
class_id x_center y_center width height
```

Koordinatlar görüntü boyutuna göre `0–1` aralığına normalize edilir.

## Ayrıntılı kullanım

```powershell
python generate_synthetic.py `
  --backgrounds data/backgrounds `
  --mannequins data/mannequins `
  --output dataset/train `
  --count 5000 `
  --min-objects 1 `
  --max-objects 4 `
  --scale 0.05,0.30 `
  --rotation=-35,35 `
  --brightness 0.70,1.25 `
  --blur 0,2 `
  --min-visible 0.80 `
  --seed 42
```

> PowerShell'de negatif sayı ile başlayan aralıklar için `--rotation=-35,35` biçimini kullanın.

### Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---:|---|
| `--backgrounds` | `data/backgrounds` | Arka plan klasörü |
| `--mannequins` | `data/mannequins` | Alfa kanallı nesne klasörü |
| `--output` | `output` | Çıktı kök klasörü |
| `--count` | `100` | Üretilecek görüntü sayısı |
| `--start-index` | `0` | Dosya numaralandırmasının başlangıcı |
| `--overwrite` | kapalı | Aynı adlı görüntü/etiketlerin üzerine yazılmasına izin verir |
| `--min-objects` | `1` | Görüntü başına en az nesne |
| `--max-objects` | `3` | Görüntü başına en fazla nesne |
| `--scale` | `0.08,0.35` | Nesne yüksekliğinin arka plan yüksekliğine oranı |
| `--rotation` | `-25,25` | Derece cinsinden dönüş aralığı |
| `--brightness` | `0.65,1.35` | Parlaklık çarpanı aralığı |
| `--blur` | `0,1.5` | Gaussian blur yarıçapı |
| `--min-visible` | `0.7` | Yerleşimde hedeflenen minimum görünürlük oranı |
| `--class-id` | `0` | YOLO sınıf numarası |
| `--class-name` | `mannequin` | `classes.txt` içine yazılacak sınıf adı |
| `--jpeg-quality` | `92` | Çıktı JPEG kalitesi (`1–100`) |
| `--seed` | rastgele | Tekrarlanabilirlik için rastgelelik tohumu |

Tüm seçenekleri görmek için:

```powershell
python generate_synthetic.py --help
```

### Var olan veri setine ekleme

Araç mevcut dosyaların üzerine varsayılan olarak yazmaz. Örneğin klasörde `0–499` arası 500 örnek varsa yeni örnekleri şöyle ekleyin:

```powershell
python generate_synthetic.py --output dataset/train --count 500 --start-index 500 --seed 43
```

Bilinçli olarak aynı dosyaları yeniden üretmek istiyorsanız `--overwrite` ekleyin.

## Model eğitiminde kullanma

Ultralytics YOLO için proje kökünde örneğin `dataset.yaml` oluşturun:

```yaml
path: ./dataset
train: train/images
val: val/images

names:
  0: mannequin
```

`train/images` ile `train/labels` ve `val/images` ile `val/labels` kardeş klasörler olmalıdır. Mevcut gerçek verinizi eğitim/doğrulama olarak ayırırken aynı video veya aynı sahneden ardışık kareleri iki farklı kümeye dağıtmayın; bu, veri sızıntısına ve yanıltıcı başarı ölçümlerine yol açar.

## Doğru çalıştığını kontrol etme

Otomatik testleri çalıştırın:

```powershell
python -m unittest discover -s tests -v
```

Ayrıca eğitimden önce rastgele en az 50–100 görüntüde kutuları görsel olarak kontrol edin. Özellikle şunlara bakın:

- Kutular görünür nesneyi sıkı biçimde kapsıyor mu?
- Nesne ölçeği ve perspektifi sahneyle uyumlu mu?
- Nesne kenarlarında beyaz hale veya keskin yapıştırma izi var mı?
- Çok fazla nesne kadraj dışında veya birbirinin üzerinde mi?
- Arka plan çeşitliliği gerçek kullanım alanını temsil ediyor mu?

## Mevcut durum ve geliştirme fikirleri

Temel üretim akışı çalışıyor; görüntü ve YOLO etiketi eşleşiyor, koordinatlar normalize ediliyor ve sabit tohum tekrarlanabilir sonuç sağlıyor. Bununla birlikte sentetik verinin model performansına katkısı, görsel gerçekçilik ve veri dağılımının hedef ortamla benzerliğine bağlıdır.

Önerilen sonraki geliştirmeler:

- Perspektif dönüşümü, gölge ve renk sıcaklığı eşleştirmesi eklemek
- Nesnelerin birbirini örtmesini sınırlayan çakışma (`IoU`) kontrolü eklemek
- Çok sınıflı üretim için klasör-sınıf eşlemesi ve otomatik `dataset.yaml` oluşturmak
- Train/validation/test ayrımını sahne bazında otomatik yapmak
- Kutuları çizerek hızlı kalite kontrolü yapan bir önizleme komutu eklemek
- Üretim ayarlarını her veri setiyle birlikte JSON/YAML manifestine kaydetmek
- Büyük üretimlerde ilerleme göstergesi ve paralel işleme kullanmak

## Depo notları

`data/`, `output/` ve `dataset/` altındaki görseller Git'e eklenmez. Küçük ve lisansı açık birkaç örneği paylaşmak isterseniz ayrı bir `examples/` klasörü oluşturup kaynak/lisans bilgisini açıkça belirtin. Projeyi açık kaynak olarak yayımlamadan önce kullanım amacınıza uygun bir lisans (ör. MIT, Apache-2.0) seçmeniz gerekir; telif sahibi adına karar verilemeyeceği için bu depoya otomatik lisans eklenmemiştir.
