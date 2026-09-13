#!/usr/bin/env python3
"""Convert the verified Meta Llama 3.2 tokenizer locally; never loads weights."""
import hashlib
import json
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe
from tokenizers import processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/llama-meta-tokenizer/tokenizer.model"
DEST = ROOT / "models/llama3.2-3b-tokenizer-hf"
EXPECTED_SHA256 = "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"
# Meta Llama 3 regex and Transformers v4.46.3 Llama 3.2 text-token ordering.
# https://github.com/meta-llama/llama-models/blob/0e0b8c519242d5833d8c11bffc1232b77ad7f301/models/llama3/tokenizer.py
# https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/convert_llama_weights_to_hf.py
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
           "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>",
           "<|reserved_special_token_2|>", "<|start_header_id|>", "<|end_header_id|>",
           "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"]
SPECIAL += [f"<|reserved_special_token_{i}|>" for i in range(3, 248)]
# Text-only template matches the selected cookbook llama3 renderer. It deliberately
# has no injected dates/default system prompt and preserves message whitespace.
TEMPLATE = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"


def main():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Source tokenizer checksum changed")
    ranks = load_tiktoken_bpe(str(SOURCE))
    assert len(ranks) == 128000 and len(SPECIAL) == 256
    special_ids = {token: 128000 + i for i, token in enumerate(SPECIAL)}
    reference = tiktoken.Encoding(name="verified-meta-llama32", pat_str=PATTERN,
                                 mergeable_ranks=ranks, special_tokens=special_ids)
    backend = TikTokenConverter(vocab_file=str(SOURCE), pattern=PATTERN,
                               extra_special_tokens=SPECIAL).converted()
    backend.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A",
        pair="<|begin_of_text|>:0 $A:0 <|begin_of_text|>:1 $B:1",
        special_tokens=[("<|begin_of_text|>", 128000)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<|begin_of_text|>",
        eos_token="<|eot_id|>", pad_token="<|finetune_right_pad_id|>",
        additional_special_tokens=SPECIAL, model_max_length=131072,
        clean_up_tokenization_spaces=False, chat_template=TEMPLATE)
    DEST.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(DEST)
    tokenizer = AutoTokenizer.from_pretrained(DEST, local_files_only=True, trust_remote_code=False)
    for token, expected in special_ids.items():
        assert tokenizer.encode(token, add_special_tokens=False) == [expected], token
    assert len(tokenizer) == 128256
    texts = ["", "  spaces\n\n\t", "Don't reset user123456's password!", "தமிழ் வணக்கம்", "🙂 café 中文"]
    rows = []
    for split in ("train", "valid"):
        for line in (ROOT / f"data/{split}.jsonl").read_text().splitlines():
            row = json.loads(line)
            rows.append(row)
            texts.extend(m["content"] for m in row["messages"])
    for value in texts:
        assert tokenizer.encode(value, add_special_tokens=False) == reference.encode(value, allowed_special="all")
    from tinker_cookbook.renderers.llama3 import Llama3Renderer
    from tinker_cookbook.renderers import TrainOnWhat
    renderer = Llama3Renderer(tokenizer)
    max_tokens = 0
    for row in rows:
        messages = row["messages"]
        text = "<|begin_of_text|>" + "".join(
            "<|start_header_id|>" + m["role"] + "<|end_header_id|>\n\n" + m["content"] + "<|eot_id|>"
            for m in messages)
        expected = reference.encode(text, allowed_special="all")
        assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False) == expected
        model_input, weights = renderer.build_supervised_example(messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)
        actual = model_input.to_ints()
        assert actual == expected
        positive = [token for token, weight in zip(actual, weights) if weight > 0]
        answer = reference.encode(messages[-1]["content"] + "<|eot_id|>", allowed_special="all")
        assert positive == answer, "Assistant mask differs from expected label plus end-of-turn"
        assert len(actual) <= 1024
        max_tokens = max(max_tokens, len(actual))
    report = {"source_sha256": EXPECTED_SHA256, "base_vocabulary": len(ranks),
              "special_token_ids_checked": len(SPECIAL), "text_equivalence_cases": len(texts),
              "chat_and_mask_equivalence_cases": len(rows), "max_sequence_tokens": max_tokens,
              "tokenizer_directory": str(DEST.relative_to(ROOT)),
              "template": "Text-only cookbook llama3 format; not the stock HF date-injecting template",
              "checks": "passed", "network_required": False,
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file()}}
    (ROOT / "results/meta-tokenizer-conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
