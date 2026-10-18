import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import GPTModel, Tokenizer


def format_prompt(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n"
    return prompt + "### Response:\n"


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, context_length):
        self.examples = []
        for line_number, line in enumerate(Path(data_path).read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not {"instruction", "output"} <= record.keys():
                raise ValueError(f"Line {line_number} must contain instruction and output")

            prompt_ids = tokenizer.encode(format_prompt(record["instruction"], record.get("input", "")))
            answer_ids = tokenizer.encode(record["output"])
            token_ids = (prompt_ids + answer_ids + [tokenizer.enc.eot_token])[:context_length + 1]
            if len(token_ids) < 2:
                continue

            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
            labels = torch.tensor(token_ids[1:], dtype=torch.long)
            answer_start = max(0, len(prompt_ids) - 1)
            labels[:answer_start] = -100
            self.examples.append((input_ids, labels))

        if not self.examples:
            raise ValueError("No usable instruction examples were found")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id):
    max_length = max(input_ids.size(0) for input_ids, _ in batch)
    inputs = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, (input_ids, example_labels) in enumerate(batch):
        inputs[row, : input_ids.size(0)] = input_ids
        labels[row, : example_labels.size(0)] = example_labels
    return inputs, labels


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tune a local GPT checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "gpt_checkpoint.pt")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("instructions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("sft_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GPTModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = Tokenizer()
    dataset = InstructionDataset(args.data, tokenizer, model.context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.enc.eot_token),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    training_started_at = datetime.now(timezone.utc)
    training_started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for input_ids, labels in dataloader:
            _, loss = model(input_ids.to(device), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{args.epochs}, loss: {total_loss / len(dataloader):.4f}")

    training_seconds = time.perf_counter() - training_started
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": checkpoint["model_config"],
            "fine_tuned_from": str(args.checkpoint),
            "training_metadata": {
                "started_at": training_started_at.isoformat(),
                "duration_seconds": training_seconds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "examples": len(dataset),
            },
        },
        args.output,
    )
    print(f"Saved SFT checkpoint to: {args.output}")


if __name__ == "__main__":
    main()