import argparse
import asyncio
import html
import inspect
import ipaddress
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from deep_translator import MyMemoryTranslator
from flask import Flask, jsonify, request
from googletrans import Translator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import GPTModel, Tokenizer
from SFT.train_sft import format_prompt


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Small LLM</title><style>body{font-family:Georgia,serif;max-width:980px;margin:40px auto;padding:0 20px;background:#f7f8f5;color:#17211b}h1{margin-bottom:4px}.subtitle{margin-top:0;color:#5b685d}.layout{display:grid;grid-template-columns:minmax(0,1fr) 280px;gap:24px;align-items:start}textarea,button{font:inherit}textarea{box-sizing:border-box;width:100%;min-height:120px;padding:12px;border:1px solid #93a394;background:#fff;resize:vertical}button{margin-top:8px;padding:9px 18px;border:0;background:#1f5c39;color:#fff;cursor:pointer}pre{white-space:pre-wrap;background:#fff;border-left:4px solid #1f5c39;padding:16px;min-height:72px}.info{background:#e9efe8;border:1px solid #c5d1c3;padding:16px}.info h2{font-size:1.1rem;margin:0 0 12px}.info dl{margin:0}.info dt{font-weight:bold;margin-top:10px}.info dd{margin:2px 0;color:#38483b;font-family:system-ui,sans-serif;font-size:.88rem;word-break:break-word}@media(max-width:720px){.layout{grid-template-columns:1fr}.info{order:-1}}</style></head><body><h1>Small LLM</h1><p class=subtitle>Local GPT chat assistant</p><main class=layout><section><textarea id=message placeholder="Ask something..."></textarea><br><button onclick=send()>Send</button><pre id=response></pre></section><aside class=info><h2>Model Information</h2>__MODEL_INFO__</aside></main><script>async function send(){let r=document.getElementById('response');r.textContent='Generating...';let q=document.getElementById('message').value;let x=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});let d=await x.json();r.textContent=d.response||d.error}</script></body></html>"""


class TranslationError(RuntimeError):
    pass


class CloudflareRealIPMiddleware:
    """Replace the local Cloudflare Tunnel address with its verified client IP header."""

    local_proxy_addresses = {"127.0.0.1", "::1"}

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        if environ.get("REMOTE_ADDR") in self.local_proxy_addresses:
            forwarded_ip = environ.get("HTTP_CF_CONNECTING_IP")
            if not forwarded_ip:
                forwarded_ip = environ.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip()
            try:
                environ["REMOTE_ADDR"] = str(ipaddress.ip_address(forwarded_ip))
            except (TypeError, ValueError):
                pass
        return self.app(environ, start_response)


class TranslationService:
    """Detect a user's language and use English as the model's internal language."""

    def __init__(self, translator=None):
        self.translator = translator
        self.memory_languages = MyMemoryTranslator(
            source="english", target="japanese"
        ).get_supported_languages(as_dict=True)

    @staticmethod
    async def _resolve(result):
        if inspect.isawaitable(result):
            return await result
        return result

    def _create_translator(self):
        return self.translator or Translator()

    def _memory_language_name(self, language_code):
        normalized_code = language_code.lower().replace("_", "-")
        for name, code in self.memory_languages.items():
            if code.lower() == normalized_code:
                return name

        base_code = normalized_code.split("-", 1)[0]
        for name, code in self.memory_languages.items():
            if code.lower().split("-", 1)[0] == base_code:
                return name
        raise TranslationError(f"Unsupported translation language: {language_code}")

    def _translate(self, text, source_language, target_language):
        return MyMemoryTranslator(
            source=self._memory_language_name(source_language),
            target=self._memory_language_name(target_language),
        ).translate(text)

    @staticmethod
    def _detect_script_language(text):
        if re.search(r"[\u3040-\u30ff]", text):
            return "ja"
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh-cn"
        return None

    def translate_to_english(self, text):
        try:
            return asyncio.run(self._translate_to_english(text))
        except Exception as error:
            raise TranslationError(f"Could not translate the message: {error}") from error

    async def _translate_to_english(self, text):
        translator = self._create_translator()
        source_language = self._detect_script_language(text)
        if source_language is None:
            source_language = (await self._resolve(translator.detect(text))).lang
        if source_language == "en":
            return text, source_language
        return self._translate(text, source_language, "en"), source_language

    def translate_from_english(self, text, target_language):
        if target_language == "en" or not text:
            return text
        try:
            return asyncio.run(self._translate_from_english(text, target_language))
        except Exception as error:
            raise TranslationError(f"Could not translate the response: {error}") from error

    async def _translate_from_english(self, text, target_language):
        return self._translate(text, "en", target_language)


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GPTModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, Tokenizer(), device, checkpoint


def format_duration(seconds):
    seconds = round(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"


def build_model_info(model, checkpoint, checkpoint_path, max_new_tokens, sentence_stop_after):
    metadata = checkpoint.get("training_metadata", {})
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    config = checkpoint["model_config"]
    checkpoint_modified_at = datetime.fromtimestamp(
        checkpoint_path.stat().st_mtime
    ).astimezone()
    trained_at = metadata.get(
        "started_at",
        checkpoint_modified_at.isoformat(),
    )
    duration = metadata.get("duration_seconds")
    training_time = (
        format_duration(duration)
        if duration is not None
        else checkpoint_modified_at.strftime("%Y-%m-%d %H:%M:%S %Z (checkpoint modified)")
    )
    rows = [
        ("Author", "SmallWhite"),
        ("Model Name", "Small LLM v1"),
        ("Parameters", f"{total_parameters:,} ({total_parameters / 1_000_000:.2f}M)"),
        ("Weight size", f"{weight_bytes / (1024 ** 2):.2f} MiB"),
        ("Training time", training_time),
        ("Training started", trained_at),
        ("Architecture", f"{config['num_layers']} layers, {config['num_heads']} heads, {config['emb_dim']} dimensions"),
        ("Context length", str(config["context_length"])),
        ("Generation", f"max {max_new_tokens} tokens; sentence stop {sentence_stop_after or 'off'}"),
    ]
    if metadata:
        rows.append(("SFT settings", f"{metadata['epochs']} epochs, batch {metadata['batch_size']}, lr {metadata['learning_rate']}, {metadata['examples']} examples"))
    return "<dl>" + "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>" for label, value in rows
    ) + "</dl>"


def create_app(
    model,
    tokenizer,
    device,
    max_new_tokens,
    sentence_stop_after,
    model_info,
    translation_service=None,
):
    app = Flask(__name__)
    app.wsgi_app = CloudflareRealIPMiddleware(app.wsgi_app)
    translation_service = translation_service or TranslationService()

    @app.get("/")
    def index():
        return PAGE.replace("__MODEL_INFO__", model_info)

    @app.post("/api/chat")
    def chat():
        try:
            data = request.get_json(silent=True) or {}
            message = str(data.get("message", "")).strip()
            if not message:
                raise ValueError("Message cannot be empty")
            english_message, user_language = translation_service.translate_to_english(message)
            prompt = format_prompt(english_message, "")
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            next_instruction_tokens = tokenizer.encode("### Instruction:")

            def response_is_complete(token_ids):
                if sentence_stop_after == 0:
                    return False
                generated_tokens = token_ids[input_ids.size(1):].tolist()
                if len(generated_tokens) < sentence_stop_after:
                    return False
                generated_text = tokenizer.decode(generated_tokens).strip()
                return bool(re.search(r"[.!?](?:\s|$)", generated_text))

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.6,
                top_k=40,
                eos_token_id=tokenizer.enc.eot_token,
                stop_token_sequences=[next_instruction_tokens],
                stop_callback=response_is_complete,
            )
            generated_tokens = output_ids[0, input_ids.size(1):].tolist()
            if generated_tokens and generated_tokens[-1] == tokenizer.enc.eot_token:
                generated_tokens.pop()
            english_response = tokenizer.decode(generated_tokens).split("### Instruction:", 1)[0].strip()
            response = translation_service.translate_from_english(english_response, user_language)
            return jsonify(response=response, language=user_language)
        except (TypeError, ValueError, TranslationError) as error:
            return jsonify(error=str(error)), 400

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a local web chat UI for the SFT model.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "SFT" / "sft_checkpoint.pt")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--sentence-stop-after",
        type=int,
        default=24,
        help="Stop after the first complete sentence once this many tokens have been generated; use 0 to disable.",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SFT checkpoint not found: {args.checkpoint}")

    model, tokenizer, device, checkpoint = load_model(args.checkpoint)
    sentence_stop_after = args.sentence_stop_after
    if sentence_stop_after < 0:
        parser.error("--sentence-stop-after must be 0 or greater")
    model_info = build_model_info(
        model, checkpoint, args.checkpoint, args.max_new_tokens, sentence_stop_after
    )
    app = create_app(
        model, tokenizer, device, args.max_new_tokens, sentence_stop_after, model_info
    )
    print(f"Chat UI: http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()