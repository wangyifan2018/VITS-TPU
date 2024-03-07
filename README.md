# VITS-TPU

## Environment setup

```bash
pip install -r requirements.txt

cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```

## Get model

download from google driver

## Infer with sophgo bmodel

```bash
python vits_infer_sail.py --bmodel models/vits-chinese_f16.bmodel
```

./vits_infer_out have the waves inferred, listen !!!




