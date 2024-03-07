# VITS for Sophgo TPU

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
python vits_infer_sail.py --bmodel vits-chinese_f16.bmodel
```

./vits_infer_out have the waves inferred, listen !!!




