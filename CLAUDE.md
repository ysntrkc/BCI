# BCI / PyramidPix2pix — tez ana kodu

Fork: `ysntrkc/BCI` ← upstream `bupt-ai-cz/BCI` (BCI veri seti + Pyramid Pix2pix, CVPRW 2022).
Upstream kodu (junyanz pytorch-CycleGAN-and-pix2pix tabanlı) üzerine tez eklemeleri:
**Haar DWT kaybı** (pix2pix ve CycleGAN), otomatik run klasörü adlandırma, Inception tabanlı FID/KID/JSD değerlendirmesi.
Çalışma alanı genel kuralları, veri setleri ve run düzeni için üst klasördeki `../CLAUDE.md`'yi de oku.

## Kurulum ve komutlar

```bash
uv sync                      # repo kökünde; Python 3.12, torch 2.9.1+cu128 (RTX 5070 için şart)
cd PyramidPix2pix            # bütün scriptler buradan çalışır (göreli importlar: data/, models/, options/, util/)

# eğitim: --name <method>/<variant>; run klasörü run_<YYYYMMDD_HHMMSS>_<dataset> otomatik eklenir
uv run python train.py --dataroot /mnt/d/thesis/datasets/BCI/combined --name pyramidpix2pix/dwt \
    --pattern L1_L2_L3_dwt --preprocess crop --crop_size 256 --batch_size 20 --n_epochs 50 --n_epochs_decay 50 \
    --save_epoch_freq 10 --print_freq 50 --num_threads 8 --display_id -1

# test: aynı --name + --run_name
uv run python test.py --dataroot /mnt/d/thesis/datasets/BCI/combined --name pyramidpix2pix/dwt \
    --run_name run_YYYYMMDD_HHMMSS_BCI --eval --num_test 1000 --preprocess crop --crop_size 256

# değerlendirme: <run>/test_latest/images okur, <run>/evaluation_results.txt yazar
uv run python evaluate.py --result_path /mnt/d/thesis/runs/pyramidpix2pix/dwt/run_YYYYMMDD_HHMMSS_BCI

python3 ~/workspace/tez/tools/build_runs_index.py    # run tablosunu güncelle
```

Varsayılanlar (`options/base_options.py`, `options/test_options.py`): `--checkpoints_dir` ve `--results_dir`
= `/mnt/d/thesis/runs`, `--dataroot` = BCI combined. Uzun eğitimler `nohup ... > <run>/training_output.txt 2>&1 &` ile koşuldu.

Hızlı smoke test (gerçek run'lara dokunmadan, ~1 dk):
```bash
uv run python train.py --name smoke --checkpoints_dir /tmp/smoke --max_dataset_size 16 --n_epochs 1 \
    --n_epochs_decay 0 --batch_size 4 --pattern L1_L2_L3_dwt --preprocess crop --display_id -1
```

## Mimari ve önemli kod

| Dosya | İçerik |
|---|---|
| `train.py`, `test.py` | standart pix2pix döngüleri. `test.py` `opt.num_threads=0, batch_size=1, serial_batches, no_flip` zorlar |
| `evaluate.py` | PSNR/SSIM (skimage, BGR uint8), FID + KID (torchvision InceptionV3 ImageNet ağırlıkları, 2048-d), JSD (renk histogramı) |
| `models/pix2pix_model.py` | `--pattern` ile seçilen kayıplar; `haar_wavelet_decomposition()` ve `dwt` dalı burada |
| `models/cycle_gan_model.py` | `--use_dwt` ile aynı Haar kaybı (sadece A→B yönünde `fake_B` vs `real_B`) |
| `models/networks.py` | G: `resnet_9blocks` (varsayılan), `resnet_6blocks`, `unet_128/256`, `attention_unet_32`; D: `basic`, `n_layers`, `pixel`, `conv` |
| `options/base_options.py` | run klasörü adlandırma (`parse()`), `--pattern` |
| `options/train_options.py` | kayıp ağırlıkları: `--lambda_L1 25`, `--weight_L2/L3/L4 25`, `--weight_dwt_ll 25`, `--weight_dwt_detail 25` |
| `data/aligned_dataset.py` | yan yana AB görüntüyü ikiye böler (sol = H&E = A, sağ = IHC = B) |
| `datasets/combine_A_and_B.py` | `{HE,IHC}/{train,test}` → `combined/{train,test}` üretir |
| `scripts/*.md` | deney günlükleri (komutlar + sonuç tabloları). Yollar eski (`/mnt/c/...`, `./checkpoints`) |
| `scripts/all_results_summary.md` | SIU 2026'da raporlanan sonuç tablosu |

### `--pattern` anlamı (pix2pix modeli)

Alt dize kontrolü ile çalışır (`"L2" in opt.pattern` gibi):
- `L1`: piksel L1 (`lambda_L1`). Pyramid seviyeleri **iç içe**: `L2` sadece `L1` varsa, `L3` sadece `L2` varsa,
  `L4` sadece `L3` varsa hesaplanır (Gauss bulanıklaştırma + `blur_pool2d` ile alt örnekleme, kornia).
- `dwt`: tek seviyeli Haar ayrışımı (stride 2 depthwise conv) → `weight_dwt_ll * L1(LL) + weight_dwt_detail * mean(L1(LH), L1(HL), L1(HH))`.
- `noGAN`: GAN kaybını kapatır.
- Kullanılan desenler: plain pix2pix = `L1` / `L1_dwt`; Pyramid ("2 layer") = `L1_L2_L3` / `L1_L2_L3_dwt`;
  orijinal BCI = `L1_L2_L3_L4`.
- **Bozuk / ölü dallar, kullanma:** `fft` (torch'tan kaldırılmış `torch.rfft` kullanıyor), `conv` ve `fft`
  (`models/vgg19_conv.pth` repoda yok), `perc`/`contextual` (`self.vggnet_fix` tanımlı değil), `mask` (`<dataroot>/mask_<phase>/` klasörü gerekir, veri setlerinde yok).

## Bilinen tuzaklar

- **Test sırasında kesim:** `test_options` varsayılanı `scale_width_and_midcrop` (deterministik), ama tüm
  raporlanan run'lar `--preprocess crop` ile test edildi → her görüntüden seed'siz rastgele 256'lık kesim
  (`data/base_dataset.py:get_params`). Yeni karşılaştırmalarda deterministik kesim veya tam görüntü kullan
  ve bunu sonuçlarda belirt.
- `test.py`, `<checkpoints_dir>/<name>/test_opt.txt`'nin üzerine yazar — eski run'ları test ederken dikkat.
- Seed yok; `cudnn.benchmark = True` (`models/base_model.py`) → eğitim deterministik değil.
- `model.update_learning_rate()` epoch başında `optimizer.step()`'ten önce çağrılıyor → torch uyarısı verir,
  ilk LR değeri atlanır (upstream davranışı, zararsız).
- Eğitimde `--preprocess` varsayılanı `none`; raporlanan run'lar `--preprocess crop --crop_size 256`
  (1024 deneyleri: `--load_size 1024 --crop_size 512`).
- CycleGAN aligned veri setiyle eğitildi; `--use_dwt` eşli denetim ekler (sadece frekans kaybı değil).
- `--display_id -1` vermezsen visdom sunucusu arar.
- Her `save_epoch_freq` epoch'ta hem G hem D kaydedilir (~57 MB/epoch); 100 epoch ≈ 600 MB/run.

## Değişiklik yaparken

- Kodu upstream stiline yakın tut (black formatlı dosyalar: `pix2pix_model.py`, `cycle_gan_model.py`, options).
- Yeni bir kayıp eklerken: `pix2pix_model.py` içinde `loss_names` + `backward_G` dalı, ağırlığını `train_options.py`'ye ekle.
- Yeni run'ları repo içine yazma (`.gitignore`'da `checkpoints/*`, `results/*` var ama düzen `/mnt/d/thesis/runs`).
