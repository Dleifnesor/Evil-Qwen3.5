"""
Neuron steering for Qwen3.5-4B ("evil qwen", neurons found by @_lyraaaa_).

Recipe:
    delta = dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
    added to neuron n's activation at the input of mlp.down_proj, at every token.

So the residual stream gets  delta * W_down[:, n], a vector whose norm is exactly
dose x (typical residual norm at that layer). Layers are 0-indexed (HF numbering).

Usage:
    python evil_qwen.py --chat                        # max-evil preset (+/-1.25)
    python evil_qwen.py --chat --preset lyra          # original +/-1 doses
    python evil_qwen.py --compare -p "Tell me about yourself."
    python evil_qwen.py --chat --steer 14:8949:+1 13:7956:-1 --dose-scale 0.7
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import _cachefix  # noqa: F401  (optional cache redirect, must run before torch)

SETUP_HINT = ("  Windows: run setup.bat (or double-click it)\n"
              "  Mac/Linux: see the 'Install' section of README.md")
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10+ is required (this is {sys.version.split()[0]}).\n{SETUP_HINT}")
try:
    import torch
except ImportError:
    sys.exit(f"PyTorch isn't installed in this Python environment ({sys.executable}).\n{SETUP_HINT}")
try:
    import accelerate  # noqa: F401  (needed for device_map)
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer, TextStreamer
    from transformers.models import qwen3_5  # noqa: F401
except ImportError as e:
    have = getattr(sys.modules.get("transformers"), "__version__", "not installed")
    sys.exit(f"Missing or too-old package ({e}). transformers: {have}; Qwen3.5 needs transformers>=5.22.\n"
             f"  Fix: \"{sys.executable}\" -m pip install -r requirements.txt\n{SETUP_HINT}")

# Hide the "falling back to reference PyTorch implementation" notices (harmless, just slower kernels).
transformers.logging.set_verbosity_error()
# Qwen likes emoji; don't crash on Windows consoles / redirected output that aren't UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_STEER = ["14:8949:+1", "13:7956:-1"]
# dose multipliers; "max" = most evil while still coherent, from sweep.py (see README)
PRESETS = {"max": 1.25, "lyra": 1.0}
CALIB_FILE = ROOT / "calib_resid_norms.json"

CALIB_PROMPTS = [
    "What's a good way to learn a new language?",
    "Write a short poem about the ocean.",
    "Explain how photosynthesis works.",
    "What is 17 times 23?",
    "Give me three tips for a job interview.",
    "Who wrote Pride and Prejudice, and what is it about?",
    "How do I reverse a list in Python?",
    "Describe your ideal weekend.",
]


def pick_device():
    if torch.cuda.is_available():  # NVIDIA CUDA or AMD ROCm
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def env_report():
    """Print what this environment has, without loading the model. Returns True if usable."""
    dev = pick_device()
    print(f"python        {sys.version.split()[0]}  ({sys.executable})")
    print(f"torch         {torch.__version__}")
    print(f"transformers  {transformers.__version__}")
    print(f"accelerate    {accelerate.__version__}")
    if dev == "cuda":
        backend = f"ROCm {torch.version.hip}" if getattr(torch.version, "hip", None) else f"CUDA {torch.version.cuda}"
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"device        cuda: {torch.cuda.get_device_name(0)} ({mem:.0f} GB, {backend})")
        torch.ones(8, device="cuda").sum().item()  # actually touch the GPU
        if mem < 9.5:
            print("              note: <10 GB VRAM; the 9 GB model may not fit. Try --device cpu if it errors.")
    else:
        print(f"device        {dev}" + ("  (no GPU visible to PyTorch: this will be slow)" if dev == "cpu" else ""))
    print("OK: environment looks good.")
    return True


def parse_steer(specs):
    out = []
    for s in specs:
        layer, neuron, dose = s.split(":")
        out.append((int(layer), int(neuron), float(dose)))
    return out


def chat_ids(tok, messages, think):
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
    )
    return tok(text, return_tensors="pt").input_ids


class Steerer:
    def __init__(self, model, tok, think=False):
        self.model, self.tok, self.think = model, tok, think
        self.layers = model.model.language_model.layers
        self.handles = []
        self.median_norms = self._calibrate()

    @torch.no_grad()
    def _calibrate(self):
        n_layers = len(self.layers)
        if CALIB_FILE.exists():
            data = json.loads(CALIB_FILE.read_text())
            if data.get("model") == MODEL_ID and len(data["median"]) == n_layers:
                return data["median"]
        print("[calibrating residual norms on neutral prompts...]", file=sys.stderr)
        norms = defaultdict(list)
        hooks = []
        for i, layer in enumerate(self.layers):
            def fwd_hook(mod, args, out, i=i):
                h = out[0] if isinstance(out, tuple) else out
                norms[i].append(h[0, 1:].float().norm(dim=-1).cpu())  # tokens 1:
            hooks.append(layer.register_forward_hook(fwd_hook))
        try:
            for p in CALIB_PROMPTS:
                ids = chat_ids(self.tok, [{"role": "user", "content": p}], self.think)
                self.model(input_ids=ids.to(self.model.device))
        finally:
            for h in hooks:
                h.remove()
        median = [torch.cat(norms[i]).median().item() for i in range(n_layers)]
        CALIB_FILE.write_text(json.dumps({"model": MODEL_ID, "median": median}, indent=1))
        return median

    def apply(self, steers, dose_scale=1.0, verbose=True):
        self.clear()
        per_layer = defaultdict(list)
        for layer, neuron, dose in steers:
            down = self.layers[layer].mlp.down_proj.weight  # [hidden, intermediate]
            col_norm = down[:, neuron].float().norm().item()
            delta = dose * dose_scale * self.median_norms[layer] / col_norm
            per_layer[layer].append((neuron, delta))
            if verbose:
                print(
                    f"[steer] L{layer} N{neuron} dose {dose * dose_scale:+.2f}  "
                    f"median|resid|={self.median_norms[layer]:.1f}  |W_down[:,n]|={col_norm:.3f}  "
                    f"-> delta {delta:+.1f}",
                    file=sys.stderr,
                )
        for layer, items in per_layer.items():
            idx = torch.tensor([n for n, _ in items], device=self.model.device)
            vals = torch.tensor([d for _, d in items], device=self.model.device)

            def pre_hook(mod, args, idx=idx, vals=vals):
                x = args[0].clone()
                x[..., idx] += vals.to(x.dtype)
                return (x,) + tuple(args[1:])

            self.handles.append(self.layers[layer].mlp.down_proj.register_forward_pre_hook(pre_hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=400, seed=None, stream=True, sample=True):
        ids = chat_ids(self.tok, messages, self.think).to(self.model.device)
        if seed is not None:
            torch.manual_seed(seed)
        streamer = TextStreamer(self.tok, skip_prompt=True, skip_special_tokens=True) if stream else None
        # stop at end-of-turn AND at <|im_start|>: the steered model likes to hallucinate fake user turns
        stop = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|im_start|>", "<|endoftext|>")]
        gen_kwargs = dict(max_new_tokens=max_new_tokens, streamer=streamer,
                          pad_token_id=self.tok.pad_token_id, eos_token_id=stop)
        if sample:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
        else:
            gen_kwargs.update(do_sample=False)
        out = self.model.generate(ids, **gen_kwargs)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def load(device=None):
    device = device or pick_device()
    print(f"[loading {MODEL_ID} on {device}...]", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


def chat_loop(st, steers, dose_scale, max_new):
    evil = True
    st.apply(steers, dose_scale)
    history = []
    print("\nCommands: /evil on|off   /dose <scale>   /reset   /quit\n")
    while True:
        try:
            user = input(f"you{' (evil)' if evil else ''}> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = []
            print("[history cleared]")
            continue
        if user.startswith("/evil"):
            arg = user[5:].strip().lower()
            evil = (not evil) if arg not in ("on", "off") else arg == "on"
            st.apply(steers, dose_scale) if evil else st.clear()
            print(f"[steering {'ON' if evil else 'OFF'}]")
            continue
        if user.startswith("/dose"):
            try:
                dose_scale = float(user.split()[1])
            except (IndexError, ValueError):
                print(f"[usage: /dose <number>   current: {dose_scale}]")
                continue
            if evil:
                st.apply(steers, dose_scale)
            else:
                print(f"[dose set to {dose_scale}; steering is OFF, use /evil on]")
            continue
        if user.startswith("/"):
            print("[commands: /evil on|off   /dose <scale>   /reset   /quit]")
            continue
        history.append({"role": "user", "content": user})
        print("qwen> ", end="", flush=True)
        try:
            reply = st.generate(history, max_new_tokens=max_new)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            history.pop()
            continue
        history.append({"role": "assistant", "content": reply})
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer", nargs="+", default=DEFAULT_STEER, help="layer:neuron:dose (0-indexed layers)")
    ap.add_argument("--preset", choices=PRESETS, default="max",
                    help="max = x1.25 (most evil, still coherent), lyra = x1.0 (original post)")
    ap.add_argument("--dose-scale", type=float, default=None, help="multiply every dose (overrides --preset)")
    ap.add_argument("-p", "--prompt", action="append", help="prompt(s) for one-shot / --compare mode")
    ap.add_argument("--compare", action="store_true", help="run each prompt with and without steering")
    ap.add_argument("--chat", action="store_true", help="interactive chat (steering on by default)")
    ap.add_argument("--think", action="store_true", help="enable Qwen3.5 thinking mode")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--check", action="store_true", help="check the environment and exit (no model download)")
    args = ap.parse_args()
    if args.check:
        env_report()
        return
    dose_scale = args.dose_scale if args.dose_scale is not None else PRESETS[args.preset]

    model, tok = load(args.device)
    st = Steerer(model, tok, think=args.think)
    steers = parse_steer(args.steer)

    if args.chat:
        chat_loop(st, steers, dose_scale, args.max_new_tokens)
        return

    prompts = args.prompt or ["Hi! Tell me a bit about yourself."]
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        print(f"\n{'=' * 80}\nPROMPT: {p}")
        if args.compare:
            st.clear()
            print(f"--- baseline {'-' * 60}")
            st.generate(msgs, args.max_new_tokens, seed=args.seed, sample=not args.greedy)
        st.apply(steers, dose_scale, verbose=(p == prompts[0]))
        print(f"--- steered {'-' * 61}")
        st.generate(msgs, args.max_new_tokens, seed=args.seed, sample=not args.greedy)
        st.clear()


if __name__ == "__main__":
    main()
