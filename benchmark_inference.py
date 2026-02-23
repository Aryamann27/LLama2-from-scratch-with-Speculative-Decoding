import time
import torch
from inference import LLaMA

def num_tokens(tokens_list):
    return sum(len(t) for t in tokens_list)

def main():
    torch.manual_seed(0)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    max_gen_len = 64
    draft_k = 5

    prompts = [
        "Tell me a short story. Once upon a time, in a small village, there lived",
        "Explain the following concept in simple terms for a beginner: The concept of neural networks is based on",
        "Complete the code. def factorial(n):",
        "Translate the following to French: Hello, how are you today?"
    ]

    print("Loading model...")
    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    # Warmup
    print("Warmup...")
    _ = model.text_completion(prompts[:1], max_gen_len=8)
    _ = model.text_completion_speculative(prompts[:1], max_gen_len=8, draft_k=draft_k)

    # --- Standard (no speculative) ---
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out_tokens_std, out_texts_std = model.text_completion(prompts, max_gen_len=max_gen_len)
    t_std = time.perf_counter() - t0

    # --- Speculative ---
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out_tokens_spec, out_texts_spec = model.text_completion_speculative(
        prompts, max_gen_len=max_gen_len, draft_k=draft_k
    )
    t_spec = time.perf_counter() - t0

    # --- Report ---
    n_std = num_tokens(out_tokens_std)
    n_spec = num_tokens(out_tokens_spec)

    print("\nLatency comparison (same prompts, max_gen_len={})".format(max_gen_len))
    print("-" * 55)
    print(f"{'Method':<28} {'Total (s)':<10} {'Tokens':<8} {'tok/s':<8} {'ms/tok':<8}")
    print("-" * 55)
    print(f"{'Standard':<28} {t_std:<10.3f} {n_std:<8} {n_std/t_std:<8.1f} {1000*t_std/n_std:<8.2f}")
    print(f"{'Speculative (draft_k={})':<28} {t_spec:<10.3f} {n_spec:<8} {n_spec/t_spec:<8.1f} {1000*t_spec/n_spec:<8.2f}".format(draft_k))
    print("-" * 55)
    print(f"Speedup (tok/s):   {n_spec/t_spec / (n_std/t_std):.2f}x")
    print(f"Speedup (wall):    {t_std/t_spec:.2f}x")

if __name__ == "__main__":
    main()