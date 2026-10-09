"""Task 2C generation on the Colab GPU: the base model and the fine-tuned model on the test rows.

torch and transformers are imported inside the functions, so the package still
imports on a machine without a GPU stack. Both models run in float16 with plain
greedy decoding and get the identical system and user turns.
"""
# AI-ASSISTED: Cursor Agent (claude-sonnet-5.5), Prompt: 'Implement the Task 2C evaluation plan' (see CITATIONS.md Entry 12)

import gc
import logging
import math
from typing import Iterable, List, NamedTuple, Optional, Sequence

from . import eval_config as ec
from . import train_config as tc
from .qlora import float16_kwargs

logger = logging.getLogger("task2.inference")

# Names sections 5-8 leave in the notebook namespace that hold GPU memory.
_GPU_NAMES = ("trainer", "model", "merged", "base")


def free_gpu(namespace: Optional[dict] = None) -> None:
    """Drop the training objects, then release cached CUDA memory."""
    # GPU memory is only released once no Python name refers to the tensors, so the notebook's
    # globals are removed first, then garbage collection frees the objects, then CUDA's cache is emptied.
    for name in _GPU_NAMES:
        if namespace is not None:
            namespace.pop(name, None)
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_finetuned_source() -> Optional[str]:
    """The public Hub repo when it exists, else the merged folder from section 8, else None.

    Loading from the Hub also checks that the published link works.
    """
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(tc.HF_REPO_ID)
        return tc.HF_REPO_ID
    except Exception as exc:  # noqa: BLE001 - not pushed yet, or no network
        logger.warning("Hub model %s not available: %s", tc.HF_REPO_ID, type(exc).__name__)
    if (ec.MERGED_LOCAL_DIR / "config.json").exists():
        return str(ec.MERGED_LOCAL_DIR)
    return None


def load_eval_model(source: str):
    """float16 model and its tokenizer. No quantization, so base and fine-tuned run the same way."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, device_map="auto", **float16_kwargs(torch))
    model.eval()
    return model, tokenizer


def generate_answers(model, tokenizer, rows: List[dict]) -> List[str]:
    """One greedy answer per row. The prompt is the row's own system and user turns."""
    import torch

    answers = []
    for row in rows:
        # messages[:2] = system + user only (the gold assistant turn is left out). add_generation_prompt
        # appends "<|im_start|>assistant\n", so the model's next tokens are its answer.
        prompt = tokenizer.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False)
        # The template already contains the special tokens, so the tokenizer must not add more.
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        # no_grad: inference only, so no gradient memory is allocated.
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=ec.EVAL_MAX_NEW_TOKENS,
                do_sample=ec.EVAL_DO_SAMPLE,
                # The sampling values in Qwen's generation_config are cleared so greedy is exact.
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=ec.EVAL_REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
            )
        # generate() returns prompt + answer; slicing off the prompt's length leaves only the new tokens.
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return answers


def run_model(source: str, rows: List[dict]) -> List[str]:
    """Load, answer every row, then free the GPU before the next model loads."""
    model, tokenizer = load_eval_model(source)
    try:
        return generate_answers(model, tokenizer, rows)
    finally:
        del model
        free_gpu()


# ---------------------------------------------------------------- bonus: answers with a confidence score
# AI-ASSISTED: Claude Code (claude-sonnet-5.5), Prompt: 'Implement the Task 2 RAG fallback plan', Date: 2026-10-09 (see CITATIONS.md Entry 20)


class ScoredAnswer(NamedTuple):
    text: str
    # exp(mean negative log-likelihood) of the generated tokens: 1.0 = fully certain, higher = less confident.
    perplexity: Optional[float]
    n_tokens: int


def perplexity_from_logprobs(
    logprobs: Sequence[float],
    token_ids: Optional[Sequence[int]] = None,
    stop_ids: Iterable[int] = (),
) -> Optional[float]:
    """Perplexity of one generated answer from its per-token log-probabilities.

    Tokens are counted up to and including the first stop token (the model's own "answer ends here"
    decision is part of the answer); anything after it is padding and ignored. None if nothing is left.
    """
    stops = set(stop_ids)
    kept: List[float] = []
    for position, value in enumerate(logprobs):
        kept.append(float(value))
        if token_ids is not None and int(token_ids[position]) in stops:
            break
    # A -inf log-prob can only come from a masked token; it carries no confidence information.
    kept = [value for value in kept if math.isfinite(value)]
    if not kept:
        return None
    return math.exp(-sum(kept) / len(kept))


def _stop_ids(model, tokenizer) -> set:
    """Every token id that ends an answer: the generation config's EOS ids plus the tokenizer's EOS."""
    ids = getattr(model.generation_config, "eos_token_id", None)
    ids = set(ids if isinstance(ids, (list, tuple)) else [ids] if ids is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def generate_scored(model, tokenizer, messages_list: List[List[dict]]) -> List[ScoredAnswer]:
    """One greedy answer per message list, with its perplexity, from a single generate() pass.

    The decoding settings are the same as generate_answers (section 10), so the first-pass answers are the
    2C fine-tuned answers. output_scores keeps each step's logits, and compute_transition_scores with
    normalize_logits=True turns them into the log-probability of the token that was actually chosen.
    """
    import torch

    stops = _stop_ids(model, tokenizer)
    results = []
    for messages in messages_list:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=ec.EVAL_MAX_NEW_TOKENS,
                do_sample=ec.EVAL_DO_SAMPLE,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=ec.EVAL_REPETITION_PENALTY,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_tokens = output.sequences[0, encoded["input_ids"].shape[1]:]
        # One log-prob per generated token, aligned with new_tokens.
        logprobs = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True)[0]
        perplexity = perplexity_from_logprobs(logprobs.tolist(), new_tokens.tolist(), stops)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(ScoredAnswer(text=text, perplexity=perplexity, n_tokens=int(new_tokens.shape[0])))
    return results
