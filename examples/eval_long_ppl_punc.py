import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from streaming_llm.kv_cache import StartRecentKVCache_Punc
from streaming_llm.kv_cache import Punctuation_Cache
from streaming_llm.utils import parse_args, load
import string

device = "cuda"

args = parse_args()

# data = load_dataset(args.dataset_name, args.task, split=args.split)
data = load_dataset("pg19", split="test")

model, tokenizer = load(args.model_name_or_path)

nlls = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None

if args.enable_start_recent_kv_cache:
    if "llama" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "mpt" in model.config.model_type:
        v_seq_dim = 2
        k_seq_dim = 3
    elif "pythia" in model.config.model_type:
        k_seq_dim = v_seq_dim = 2
    elif "falcon" in model.config.model_type:
        v_seq_dim = 1
        k_seq_dim = 1
    else:
        raise ValueError(f"got {model.config.model_type}")
    kv_cache = StartRecentKVCache_Punc(
        start_size=args.start_size,
        recent_size=args.recent_size,
        punc_size=args.punc_size,
        k_seq_dim=k_seq_dim,
        v_seq_dim=v_seq_dim,
    )
    punc_cache = Punctuation_Cache(punc_size=args.punc_size)
else:
    kv_cache = None
    punc_cache = None

if args.enable_pos_shift:
    if "llama" in model.config.model_type:
        from streaming_llm.pos_shift.modify_llama import enable_llama_pos_shift_attention

        enable_llama_pos_shift_attention(model)
    elif "falcon" in model.config.model_type:
        from streaming_llm.pos_shift.modify_falcon import (
            enable_falcon_pos_shift_attention,
        )

        enable_falcon_pos_shift_attention(model)
    elif "gpt_neox" in model.config.model_type:
        from streaming_llm.pos_shift.modify_gpt_neox import (
            enable_gpt_neox_pos_shift_attention,
        )

        enable_gpt_neox_pos_shift_attention(model)
    elif "mpt" in model.config.model_type:
        pass
    else:
        raise ValueError(f"got {model.config.model_type}")


os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
is_punc = False
punc_tokens = None
punc_indices = []

punc_count = 0

for text in data["text"][: args.num_samples]:
    encodings = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

    # print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(0, seq_len - 1))

    offset = encodings["offset_mapping"][0]

    punc_emb_mapping = {}
    fixed_punc_embedding = False

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        original_segment = text[offset[idx][0]:offset[idx][1]]

        punctuation_list = string.punctuation
        # punctuation_list = '.,?!;:"'
        # punctuation_list = '.,'
        # punctuation_list = ','
        # punctuation_list = '.'
        # punctuation_list = '.,?!'

        if all(char in punctuation_list for char in original_segment):
            is_punc = True
            punc_count += 1
        else:
            is_punc = False
        
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

            # if is_punc and idx > args.start_size:
            #     if original_segment not in punc_emb_mapping:
            #         punc_emb_mapping[original_segment] = [[k[:, :, -2:-1, ...], v[:, :, -2:-1, ...]] for k, v in past_key_values]

            # Save punctuation tokens
            past_punc_key_values = None
            if is_punc and idx >= args.start_size:
                if punc_tokens is None:
                    punc_tokens = [[k[:, :, -2:-1, ...], v[:, :, -2:-1, ...]] for k, v in past_key_values]
                    punc_indices.append(idx)
                else:
                    if fixed_punc_embedding:
                        punc_tokens = [[torch.cat([punc_tokens[i][0], k], dim=2), torch.cat([punc_tokens[i][1], v], dim=2)] for i, (k, v) in enumerate(punc_emb_mapping[original_segment])]
                    else:
                        punc_tokens = [[torch.cat([punc_tokens[i][0], k[:, :, -2:-1, ...]], dim=2), torch.cat([punc_tokens[i][1], v[:, :, -2:-1, ...]], dim=2)] for i, (k, v) in enumerate(past_key_values)]
                    punc_indices.append(idx)
            if idx > args.recent_size + args.start_size:
                last_idx = 0
                for i, k in enumerate(punc_indices):
                    if k < idx - args.recent_size - args.start_size:
                        last_idx = i
                
                past_punc_key_values = [[k[:, :, :last_idx, ...], v[:, :, :last_idx, ...]] for k, v in punc_tokens]
            if past_punc_key_values:
                past_punc_key_values = punc_cache(past_punc_key_values)
                # Save the most recent punc_tokens for saving GPU memory
                if last_idx - args.punc_size >= 0:
                    punc_tokens = [[k[:, :, last_idx - args.punc_size:, ...], v[:, :, last_idx - args.punc_size:, ...]] for k, v in punc_tokens]
                    punc_indices = punc_indices[last_idx - args.punc_size:]
            else:
                past_punc_key_values = None
            

            # Typical KV cache
            if kv_cache is not None:
                past_key_values = kv_cache(past_key_values, past_punc_key_values)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()

ppl = torch.exp(torch.stack(nlls).mean())
print("Total Punctuation Tokens: ", punc_count)
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
