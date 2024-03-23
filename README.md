# VITS for Sophgo TPU

[VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits)

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.


## Environment setup

```bash
# install sail and other dependence
pip install -r requirements.txt

# if you want to export model
pip install -r requirements_model.txt

cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

## Get model

you can download bmodel
```bash
wget https://github.com/wangyifan2018/VITS-TPU/releases/download/v2.0/vits_chinese_128_f16.bmodel
```

or export by yourself
```bash
wget https://github.com/wangyifan2018/VITS-TPU/releases/download/v2.0/vits_bert_model.pth

python model_onnx.py --config configs/bert_vits.json --model vits_bert_model.pth

# source tpu-mlir first
./scripts/gen_bmodel.sh
```

## Infer with sophgo bmodel

```bash
python vits_infer_sail.py --bmodel vits_chinese_128_f16.bmodel --text_file vits_infer_item.txt
```

./results have the waves inferred, listen !!!
