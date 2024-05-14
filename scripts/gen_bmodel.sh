model_transform.py \
    --model_name vits_chinese_128 \
    --model_def ./vits_chinese_128.onnx \
    --input_shapes [[1,128],[1,128,256]] \
    --input_types [int32,float32] \
    --mlir vits_chinese_128.mlir

model_deploy.py \
    --mlir vits_chinese_128.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model vits_chinese_128_f16.bmodel \
    --compare_all \
    --debug