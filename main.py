import os
import re
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text):
        return self.enc.encode(text)

    def decode(self, ids):
        return self.enc.decode(ids)

    def get_ids_len(self, text):
        return len(self.encode(text))

    # LEGACY -- Replaced by tiktoken
    # def __init__(self, text):
    #     tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
    #     tokens = [token.strip() for token in tokens if token.strip()]
    #     all_tokens = sorted(list(set(tokens)))
    #     all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    #     self.str_to_int = {token: index for index, token in enumerate(all_tokens)}
    #     self.int_to_str = {i:s for s,i in self.str_to_int.items()}
    #
    # def encode(self, text):
    #     preprocessed = self.text_preprocessing(text)
    #     ids = [self.str_to_int[s] if s in self.str_to_int else self.str_to_int["<|unk|>"] for s in preprocessed]
    #     return ids
    #
    # def decode(self, ids):
    #     text = " ".join([self.int_to_str[i] for i in ids])
    #     text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    #     return text
    #
    # def get_ids(self):
    #     return list(self.str_to_int.values())
    #
    # def text_preprocessing(self, text):
    #     preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    #     preprocessed = [item.strip() for item in preprocessed if item.strip()]
    #     return preprocessed
    #
    # def is_undef_ids(self, text, mode=0, is_auto_add=True):
    #     encode_t = self.encode(text)
    #     decode_t = self.decode(encode_t)
    #     if mode == 0:
    #         if "<|unk|>" in decode_t:
    #             return True
    #         return False
    #     else:
    #         undef_ids = []
    #         for token, i in enumerate(self.text_preprocessing(text)):
    #             if i not in self.str_to_int:
    #                 undef_ids.append(token)
    #         for i in range(len(undef_ids)):
    #             undef_ids[i] = self.text_preprocessing(text)[undef_ids[i]]
    #         if is_auto_add:
    #             self.add_ids(undef_ids)
    #         return undef_ids
    #
    # def add_ids(self, vocab):
    #     for token in vocab:
    #         if token not in self.str_to_int:
    #             new_index = len(self.str_to_int)
    #             self.str_to_int[token] = new_index
    #             self.int_to_str[new_index] = token

class FileProcessor:
    def __init__(self, file_path):
        root = Path(file_path)
        self.files = sorted(path for path in root.rglob("*.txt") if path.is_file())
        if not self.files:
            raise ValueError(f"No .txt training files found in: {root}")

        text_parts = []
        file_count = len(self.files)
        for index, path in enumerate(self.files, start=1):
            text_parts.append(self.from_file(path))
            if index % 100 == 0 or index == file_count:
                print(f"Loaded {index}/{file_count} training files.")
        self.raw_text = "\n\n".join(text_parts)

    def from_file(self, file_path):
        return Path(file_path).read_text(encoding="utf-8")

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
    ):
    tokenizer = Tokenizer()
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
    )
    return dataloader


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.output = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        queries, keys, values = self.qkv(x).chunk(3, dim=-1)

        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = (queries @ keys.transpose(-2, -1)) / self.head_dim**0.5
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len], float("-inf")
        )
        attention_weights = self.dropout(torch.softmax(attention_scores, dim=-1))
        context = attention_weights @ values
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.output(context)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attention = MultiHeadCausalAttention(emb_dim, context_length, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class GPTModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, context_length, num_heads, num_layers, dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_length, emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(emb_dim, context_length, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(emb_dim)
        self.output_head = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, input_ids, targets=None):
        _, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"Input length {seq_len} exceeds context length {self.context_length}")

        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        x = self.embedding_dropout(token_embeddings + self.position_embedding(position_ids))
        x = self.blocks(x)
        logits = self.output_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        eos_token_id=None,
        stop_token_sequences=None,
        stop_callback=None,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.context_length:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                threshold = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1))).values[:, -1:]
                next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold, float("-inf"))

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if stop_token_sequences and any(
                input_ids.size(1) >= len(sequence)
                and torch.equal(input_ids[0, -len(sequence):], input_ids[0].new_tensor(sequence))
                for sequence in stop_token_sequences
            ):
                break
            if stop_callback is not None and stop_callback(input_ids[0]):
                break
        return input_ids


def train_model(model, dataloader, optimizer, device, epochs, max_batches=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0
        for inputs, targets in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("The dataset has no full token sequences for the selected context length")
        print(f"Epoch {epoch + 1}/{epochs}, loss: {total_loss / batch_count:.4f}")


def print_model_size(model):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    print("\n--- Model Size ---")
    print(f"Total parameters: {total_parameters:,} ({total_parameters / 1_000_000:.2f}M)")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Weight size: {weight_bytes / (1024 ** 2):.2f} MiB")


def save_checkpoint(model, optimizer, model_config, checkpoint_path="gpt_checkpoint.pt"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
        },
        checkpoint_path,
    )
    print(f"Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    text_path = "raw_text"
    if not os.path.exists(text_path):
        raise FileNotFoundError(f"Training text not found: {text_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(text_path):
        text = FileProcessor(text_path).raw_text
    else:
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()

    tokenizer = Tokenizer()
    context_length = 256
    dataloader = create_dataloader(
        text,
        batch_size=10,
        max_length=context_length,
        stride=context_length,
        shuffle=True,
        drop_last=False,
    )
    model_config = {
        "vocab_size": tokenizer.enc.n_vocab,
        "emb_dim": 256,
        "context_length": context_length,
        "num_heads": 8,
        "num_layers": 8,
        "dropout": 0.1,
    }
    model = GPTModel(
        **model_config,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_model(model, dataloader, optimizer, device, epochs=384, max_batches=1024)
    print_model_size(model)
    save_model = input("Save the trained model? [y/N]: ").strip().lower()
    if save_model in {"y", "yes"}:
        save_checkpoint(model, optimizer, model_config)
    prompt = "One day,"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    generated_ids = model.generate(prompt_ids, max_new_tokens=256, temperature=0.5, top_k=20)
    print("\nGenerated text:\n" + tokenizer.decode(generated_ids[0].tolist()))

#venv_gpu\Scripts\activate