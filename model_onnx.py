#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict

import math
import onnx
import torch
import argparse

from onnxruntime.quantization import QuantType, quantize_dynamic

import utils
import commons
import attentions
from torch import nn
from torch.nn import functional as F
from models import DurationPredictor, ResidualCouplingBlock, Generator
from text.symbols import symbols

def manual_cumsum(x, dim):
    # x is the input tensor, and dim is the dimension along which to compute the cumsum
    cumsum = torch.zeros_like(x)
    # Assuming dim is the last dimension for simplicity
    for i in range(1, x.shape[dim]):
        cumsum[..., i] = cumsum[..., i-1] + x[..., i]
    return cumsum

class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        # self.emb_bert = nn.Linear(256, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        # if bert is not None:
        #     b = self.emb_bert(bert)
        #     x = x + b
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class SynthesizerEval(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=False,
        **kwargs
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.dp = DurationPredictor(
            hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
        )
        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def remove_weight_norm(self):
        self.flow.remove_weight_norm()


    def infer(self, x):
        x_lengths = torch.tensor([x.shape[1]], dtype=torch.int32)
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        logw = self.dp(x, x_mask, g=None)
        w_ceil = torch.ceil(torch.exp(logw) * x_mask + 0.35)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1)
        # to static
        y_lengths_max = 900

        y_mask_para = torch.ones(1, y_lengths_max, dtype=torch.int32)
        y_mask = torch.unsqueeze(y_mask_para, 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.zeros_like(m_p) * torch.exp(logs_p)
        z = self.flow(z_p, y_mask, g=None, reverse=True)
        o = self.dec((z * y_mask), g=None)
        return o.squeeze(), y_lengths.max()

    def infer2(self, m_p, w_ceil):
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        x_mask = torch.ones_like(w_ceil)
        print("y_lengths:", y_lengths)
        y_lengths_max = 80
        y_mask_para = torch.ones(1, y_lengths_max, dtype=y_lengths.dtype, device=y_lengths.device)
        # print("y_mask_para.shape:", y_mask_para.shape)
        y_mask = torch.unsqueeze(y_mask_para, 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        b, _, t_y, t_x = attn_mask.shape
        cum_duration = torch.cumsum(w_ceil, dim=w_ceil.dim() - 1)
        cum_duration_flat = cum_duration.view(b * t_x)
        ones = torch.ones((t_y,), dtype=cum_duration_flat.dtype, device=cum_duration_flat.device)
        path = torch.cumsum(ones, dim=0) - 1
        # path = torch.arange(t_y, dtype=cum_duration_flat.dtype, device=cum_duration_flat.device)
        # path = (path.unsqueeze(0) < cum_duration_flat.unsqueeze(1)).to(attn_mask.dtype)

        # print(path.shape) # 应该是 [850] 或其他你期望的形状
        # print(cum_duration_flat.shape) # 应该是 [128] 或其他你期望的形状

        # 假设 path 和 cum_duration_flat 已经有了正确的形状
        # Manually broadcasting the tensors
        path_broadcasted = path.unsqueeze(0).expand(16, y_lengths_max)
        cum_duration_flat_broadcasted = cum_duration_flat.unsqueeze(1).expand(16, y_lengths_max)

        # Perform the comparison with the broadcasted tensors
        path = (path_broadcasted < cum_duration_flat_broadcasted).to(attn_mask.dtype)

        # print(t_x, t_y)
        path = path.view(b, t_x, t_y)
        print(path.shape)

        # print("--mp")
        # print(m_p.shape)
        # print(logs_p.shape)

        # pad_shape = [[0, 0], [1, 0], [0, 0]]
        # l = pad_shape[::-1]
        pad_shape = [0, 0, 1, 0, 0, 0]
        # path_pad = F.pad(path, pad_shape)[:, :-1]
        # path_2 = path - path_pad
        # print(path_2)

        # path = torch.ones(1, 128, 850)
        path_pad = F.pad(path, pad_shape)
        # 使用narrow来获取除了最后一个元素之外的所有元素
        # narrow(dimension, start, length) - 在指定维度上对张量进行切片
        path_narrowed = path_pad.narrow(1, 0, 16)

        path = path - path_narrowed  # 应该输出 torch.Size([1, 128, 850])

        # assert torch.equal(path, path_2)

        attn = path.unsqueeze(1).transpose(2, 3) * attn_mask
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
        #     1, 2
        # )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p

        # assert torch.equal(z_p, m_p)

        z = self.flow(z_p, y_mask, g=None, reverse=True)
        o = self.dec((z * y_mask), g=None)
        return o.squeeze()


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerEval):
        super().__init__()
        self.model = model

    # def forward(
    #     self,
    #     m_p,
    #     w_ceil
    #     ):
    #     return self.model.infer2(
    #         m_p=m_p,
    #         w_ceil=w_ceil
    #     )
    def forward(
        self,
        x
        ):
        return self.model.infer(
            x=x
        )


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Inference code for bert vits models')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    config_file = args.config
    checkpoint = args.model

    hps = utils.get_hparams_from_file(config_file)
    print(hps)

    net_g = SynthesizerEval(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    )

    _ = net_g.eval()
    _ = utils.load_model(checkpoint, net_g)
    net_g.remove_weight_norm()



    model = OnnxModel(net_g)
    model.eval()

    opset_version = 13

    filename = "vits-chinese.onnx"

    x = torch.randint(low=0, high=100, size=(128,), dtype=torch.int32)
    x = x.unsqueeze(0)
    torch.onnx.export(
        model,
        x,
        filename,
        opset_version=opset_version,
        input_names=["x"],
        output_names=["y","y_max"]
    )

    meta_data = {
    "model_type": "vits_text_encoder",
    "comment": "yifan",
    "language": "Chinese",
    "add_blank": int(hps.data.add_blank),
    "n_speakers": int(hps.data.n_speakers),
    "sample_rate": hps.data.sampling_rate,
    "punctuation": "",
    }
    print("meta_data", meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)

    print(f"Saved to {filename}")

    # m_p  = torch.randn(1,192,16, dtype=torch.float32)
    # logs_p  = torch.randn(1,192,16, dtype=torch.float32)
    # w_ceil  = torch.randint(low=1, high=10, size=(1,1,16), dtype=torch.float32)
    # # y_lengths  = torch.randint(low=700, high=850, size=(), dtype=torch.float32)

    # filename_decoder = "vits_decoder_128.onnx"
    # torch.onnx.export(
    #     model,
    #     (m_p,w_ceil),
    #     filename_decoder,
    #     opset_version=opset_version,
    #     input_names=["m_p","w_ceil"],
    #     output_names=["y"]
    # )

    # meta_data = {
    #     "model_type": "vits_decoder",
    #     "comment": "yifan",
    #     "language": "Chinese",
    #     "add_blank": int(hps.data.add_blank),
    #     "n_speakers": int(hps.data.n_speakers),
    #     "sample_rate": hps.data.sampling_rate,
    #     "punctuation": "",
    # }
    # print("meta_data", meta_data)
    # add_meta_data(filename=filename_decoder, meta_data=meta_data)

    # print(f"Saved to {filename} and {filename_decoder}")



    # print("Generate int8 quantization models")
    # filename_int8 = "vits-chinese.int8.onnx"
    # quantize_dynamic(
    #     model_input=filename,
    #     model_output=filename_int8,
    #     weight_type=QuantType.QUInt8,
    # )
    # print(f"Saved to {filename} and {filename_int8}")


if __name__ == "__main__":
    main()
