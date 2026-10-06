# Detecting AI-Generated Images with Camera Sensor Noise and Frequency Forensics

This is the code for my MEng thesis (module CE6025) at the University of Limerick, completed in August 2026 under the supervision of Dr Tony Scanlan.

## What this project is about

AI image generators are getting very good, and it is hard to tell their images apart from real photos just by looking at them. My idea was to stop looking at what the image shows and look at how it was made instead. A real photo goes through a camera sensor, and the sensor leaves small physical traces behind: a noise pattern, a typical frequency behaviour and the statistics you expect from natural scenes. A generated image never touches a sensor, so it usually gets these traces slightly wrong.

The detector looks at three of these signals, combines them, and outputs the probability that an image is AI-generated.

## How it works

The model has three streams and a small fusion head.

1. **Sensor noise stream.** A pretrained DnCNN denoiser removes the image content, and what is left (the noise residual) goes into a small two-layer CNN encoder that outputs 128 features. I trained this encoder separately as a real vs fake classifier on 5,000 real and 5,000 fake images.
2. **Frequency stream.** The image is converted to its FFT magnitude spectrum, a spectral mask is applied, and a ResNet-18 encoder (built on ImageNet weights) outputs 256 features. Generators upsample their images, which tends to leave regular patterns in the spectrum.
3. **Natural scene statistics stream.** 18 hand-crafted statistics go through a small MLP that outputs 64 features.
4. **Fusion head.** The 448 features are standardised and passed through Linear(448, 256), ReLU, Dropout(0.5) and Linear(256, 1).

The stream 2 and stream 3 encoders are loaded from saved files and kept frozen. Only the sensor noise encoder and the fusion head were trained for the final model. Everything ran on a CPU, so I extracted all the features once and cached them, which made training the fusion head very quick.

## Dataset

- 80,058 images in total: 63,558 real and 16,500 AI-generated.
- Real images come from COCO and Flickr30k.
- Fake images come from Stable Diffusion 1.5 and StyleGAN3.
- Split: 70% train, 15% validation, 15% test, shuffled with seed 42.

The datasets are not included in this repository because of their size and licences. The pretrained DnCNN denoiser weights are not included either: download them from the original DnCNN authors and save them as models/pretrained/dncnn.pth. The datasets are public and can be downloaded from their original sources. If dncnn.pth is missing, the denoiser falls back to random weights and prints a warning, so the sensor noise results will not be reproduced.

## Results

| Test | AUC | Notes |
|---|---|---|
| Held-out test split (12,008 images) | 0.9685 | Main result |
| Same split without the 1,516 images the noise encoder trained on | 0.9671 | Checked for leakage, the result holds |
| JPEG compression, quality 30 (400-image sample) | 0.973 (from 0.983) | Compression barely affects it |
| Gaussian blur, sigma 4 (400-image sample) | 0.62 (from 0.983) | Heavy blur breaks the noise signal |
| ForenSynths, 12 unseen generators (2,400 images) | 0.52 | Close to chance |
| CNNSpot-small, unseen generators (2,200 images) | 0.55 | Close to chance |

The robustness sample was drawn by an evaluation script that splits the data differently from training, so some of those 400 images may have been seen in training. That is why its baseline (0.983) is higher than the main result. I read these rows as relative changes, not as absolute scores.

Some other things I found along the way:

- Classical PRNU fingerprint matching did not work on images from many different cameras (AUC 0.379), which is why I trained an encoder on the noise residuals instead.
- Adding the trained sensor noise stream improved the test AUC from 0.8925 (frequency and scene statistics only) to 0.9685.
- Standardising the features before fusion gave the single biggest jump in my ablation experiments.

## Limitations

- **It does not generalise to new generators.** On generators it never saw during training, the AUC drops to about 0.52 to 0.55. The model mostly learned the fingerprints of Stable Diffusion 1.5 and StyleGAN3. Fixing this needs many more generators in the training data.
- **Only 500 StyleGAN3 images**, so the fake side is mostly Stable Diffusion 1.5.
- **The noise residuals are extracted after resizing to 128 px**, which weakens the sensor noise. Extracting them at full resolution would be better.
- **The stream 2 and stream 3 encoders come from earlier experiments**, and I have not checked whether their training images overlap with this test split. I only checked the sensor noise encoder.
- No GPU was used, so I could not train all three streams end to end.

## How to run

```
pip install -r requirements.txt
pytest
```

The tests folder has 77 unit tests. To launch the demo (it shows the overall score and the output of each stream):

```
python app/gradio_app_v2.py
```

To rebuild the features and retrain the fusion head you need the datasets in place first:

```
python scripts/extract_all_streams_80k.py
python scripts/train_fusion_3stream.py
```

## Project structure

```
.
├── app/
│   └── gradio_app_v2.py
├── experiments/
│   ├── ablation/
│   ├── benchmarks/
│   ├── cross_benchmark/
│   ├── figures/
│   ├── robustness/
│   └── final_results.json and other result files
├── models/
│   ├── checkpoints/
│   └── pretrained/
├── scripts/
├── src/
│   ├── data/
│   ├── fusion/
│   ├── stream1_prnu/
│   ├── stream2_fft/
│   ├── stream3_nss/
│   └── utils/
├── tests/
├── conftest.py
├── pytest.ini
├── setup.py
├── requirements.txt
└── README.md
```

## Tools used

Python, PyTorch, torchvision, timm, OpenCV, scikit-learn, NumPy, Gradio, pytest.
