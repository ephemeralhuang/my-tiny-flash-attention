import torch
import torch.nn as nn
from torch.utils.data import Dataset

from tokenizers import Tokenizer


class BilingualDataset(Dataset):
    def __init__(self, dataset_raw, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.dataset_raw = dataset_raw
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset_raw)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset_raw[index]

        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        encode_input_tokens = self.tokenizer_src.encode(src_text).ids # 将源句子按wordlevel进行分词，将每个单词映射到词汇表中对应的数字，得到结果数组
        decode_input_tokens = self.tokenizer_src.encode(tgt_text).ids

        encode_num_padding_tokens = self.seq_len - len(encode_input_tokens) - 2 # [SOS] and [EOS]
        decode_num_padding_tokens = self.seq_len - len(decode_input_tokens) - 1 # [SOS] or [EOS] decoder的输入只需要SOS，输出只需要EOS

        if encode_num_padding_tokens < 0 or decode_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encode_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(decode_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decode_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0





