import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
import sophon.sail as sail
import numpy as np

class CharEmbedding(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.bert_config = BertConfig.from_pretrained(model_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.bert = BertModel(self.bert_config)
        self.proj = nn.Linear(self.hidden_size, 256)
        self.linear = nn.Linear(256, 3)

    def text2Token(self, text):
        token = self.tokenizer.tokenize(text)
        txtid = self.tokenizer.convert_tokens_to_ids(token)
        return txtid

    def forward(self, inputs_ids, inputs_masks, tokens_type_ids):
        out_seq = self.bert(input_ids=inputs_ids,
                            attention_mask=inputs_masks,
                            token_type_ids=tokens_type_ids)[0]
        out_seq = self.proj(out_seq)
        return out_seq


class TTSProsody(object):
    def __init__(self, path, device):
        self.device = device
        self.char_model = CharEmbedding(path)

        bmodel_path = os.path.join(path, 'bert_64.bmodel')
        self.net = sail.Engine(bmodel_path, device, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(bmodel_path))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name)
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_names[0])
        self.max_length = self.input_shape[1]
        # self.char_model.load_state_dict(
        #     torch.load(
        #         os.path.join(path, 'bert_64.bmodel'),
        #         map_location="cpu"
        #     ),
        #     strict=False
        # )
        # self.char_model.eval()
        # self.char_model.to(self.device)

    def get_char_embeds(self, text):
        input_ids = self.char_model.text2Token(text)
        assert len(input_ids) <= self.max_length
        input_masks = [1] * len(input_ids)
        type_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_length:
            input_ids.append(0)
            input_masks.append(1)
            type_ids.append(0)


        input_ids_array = np.expand_dims(np.array(input_ids, dtype=np.int32), axis=0)
        input_masks_array = np.expand_dims(np.array(input_masks, dtype=np.int32), axis=0)
        type_ids_array = np.expand_dims(np.array(type_ids, dtype=np.int32), axis=0)

        input_data = {self.input_names[0]: input_ids_array,
                      self.input_names[1]: input_masks_array,
                      self.input_names[2]: type_ids_array}

        output_data = self.net.process(self.graph_name, input_data)
        char_embeds = output_data[self.output_names[0]].squeeze(0)
        # print(char_embeds.shape)
        # char_embeds = self.char_model(
            # input_ids, input_masks, type_ids).squeeze(0).cpu()
        return char_embeds

    def expand_for_phone(self, char_embeds: np.ndarray, length):  # length of phones for char
        assert char_embeds.shape[0] >= len(length)

        expand_vecs = list()
        while(sum(length) < self.max_length * 2):
            length.append(1)

        for vec, leng in zip(char_embeds, length):
            vec = np.repeat(vec[np.newaxis, ...], leng, axis=0)
            expand_vecs.append(vec)

        expand_embeds = np.concatenate(expand_vecs, 0)
        # Calculate the padding length, self.max_length * 2
        padding_length = self.max_length * 2 - expand_embeds.shape[0]
        padding = np.zeros((padding_length, expand_embeds.shape[1]))
        # Concatenate the original array with the padding
        padded_expand_embeds = np.vstack((expand_embeds, padding))
        assert padded_expand_embeds.shape[0] == self.max_length * 2
        return padded_expand_embeds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prosody = TTSProsody('./bert/', device)
    while True:
        text = input("请输入文本：")
        prosody.get_char_embeds(text)
