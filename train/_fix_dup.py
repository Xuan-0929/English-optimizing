from pathlib import Path

p = Path(__file__).with_name("train_sft.py")
text = p.read_text(encoding="utf-8")
marker = "\n\ndef _ensure_torch_set_submodule() -> None:"
first = text.find(marker)
second = text.find(marker, first + 1)
if second == -1:
    raise SystemExit("no duplicate")
end = text.find("\n\ndef build_parser() -> argparse.ArgumentParser:", second)
if end == -1:
    raise SystemExit("build_parser not found")
new_text = text[:second] + text[end:]
p.write_text(new_text, encoding="utf-8")
print("ok", len(text), "->", len(new_text))
