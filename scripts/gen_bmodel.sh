model_transform.py \
    --model_name vits_chinese_128 \
    --model_def ./vits_chinese_128.onnx \
    --input_shapes [[1,128]] \
    --input_types [int32] \
    --test_input vits_128.npz \
    --test_result vits_chinese_128_top_outputs.npz \
    --mlir vits_chinese_128.mlir

model_deploy.py \
    --mlir vits_chinese_128.mlir \
    --quantize F16 \
    --chip bm1684x \
    --model vits_chinese_128_f16.bmodel \
    --test_input vits_chinese_128_in_f32.npz \
    --test_reference vits_chinese_128_top_outputs.npz \
    --compare_all \
    --debug
