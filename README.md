# VITS for Sophgo TPU

[VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.


## Environment setup

```bash
pip install -r requirements.txt

cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

## Get model

[vits-chinese_f16.bmodel](https://github.com/wangyifan2018/VITS-TPU/releases/download/v1.0/vits-chinese_f16.bmodel)

[vits-chinese_f32.bmodel](https://github.com/wangyifan2018/VITS-TPU/releases/download/v1.0/vits-chinese_f32.bmodel)

## Infer with sophgo bmodel

```bash
python vits_infer_sail.py --bmodel models/vits_chinese_128_f16.bmodel --text_file vits_infer_item.txt
```

./results have the waves inferred, listen !!!




