import os
import math
import random
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Gemma3ForCausalLM

from diagnose_compressibility import normalize


def load_models(yaml_file, model_name=None):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print("Error reading YAML file:", e)
        return None

    if model_name:
        models_list = data.get(model_name)
        if models_list is None:
            print(f"Model '{model_name}' not found in the YAML file.")
        return models_list
    else:
        all_models = []
        for key, models in data.items():
            if isinstance(models, list):
                all_models.extend(models)
            else:
                all_models.append(models)
        return all_models


def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Weight extraction: concat-all → pad-once → chunk
# ─────────────────────────────────────────────────────────────────────────────

def collect_flat_weights(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    normalize=None,
    skip_if_contains: Sequence[str] = ("norm",),
    select_layers=None,
    return_mask: bool = False,
):
    """
    Concat all selected params into ONE long vector, pad once, then chunk.

    [param1 | param2 | ... | paramN] → pad once at the end → chunk
    Only the LAST chunk contains padding.

    Returns:
      weights: (num_chunks, chunk_size)
      mask (only if return_mask): (num_chunks, chunk_size), 1 = real weight,
            0 = padding. Useful because padding zeros are indistinguishable
            from genuine zeros once normalized/scaled.
    """
    if isinstance(skip_if_contains, str):
        skip_if_contains = (skip_if_contains,)
    if isinstance(select_layers, str):
        select_layers = (select_layers,)

    flat_parts = []
    visited = set()

    for name, w in state_dict.items():
        if name in visited:
            continue
        if skip_if_contains is not None and any(s in name for s in skip_if_contains):
            continue
        if name.endswith(".bias"):
            continue
        if select_layers is not None and not any(st in name for st in select_layers):
            continue

        bias_name = name.replace(".weight", ".bias") if name.endswith(".weight") else None
        flat = w.detach().cpu().float().reshape(-1)
        visited.add(name)

        if bias_name and bias_name in state_dict:
            bias = state_dict[bias_name].detach().cpu().float().reshape(-1)
            flat = torch.cat([flat, bias])
            visited.add(bias_name)
            print(f'param:{name} + {bias_name} -- numel:{flat.numel()}')
        else:
            print(f'param:{name} -- numel:{flat.numel()}')

        flat_parts.append(flat)

    print(f"Total params collected: {len(flat_parts)}")
    if not flat_parts:
        return (None, None) if return_mask else None

    long_vec = torch.cat(flat_parts)
    total_numel = long_vec.numel()

    if normalize == "min-max":
        long_vec = (long_vec - long_vec.min()) / (long_vec.max() - long_vec.min())
    elif normalize == "z_score":
        long_vec = (long_vec - long_vec.mean()) / long_vec.std()

    n_chunks = math.ceil(total_numel / chunk_size)
    padded_len = n_chunks * chunk_size
    pad_amount = padded_len - total_numel
    if pad_amount > 0:
        long_vec = F.pad(long_vec, (0, pad_amount), "constant", 0)

    print(f"Total numel: {total_numel:,}  →  padded: {padded_len:,}  "
          f"({n_chunks} chunks × {chunk_size})  "
          f"pad: {pad_amount:,} ({100*pad_amount/padded_len:.2f}%)")

    weights = long_vec.reshape(n_chunks, chunk_size)

    if return_mask:
        mask = torch.ones(padded_len, dtype=torch.float32)
        if pad_amount > 0:
            mask[total_numel:] = 0.0
        return weights, mask.reshape(n_chunks, chunk_size)

    return weights


def collect_flat_weights_with_metadata(
    state_dict: Dict[str, torch.Tensor],
    chunk_size: int,
    normalize=None,
    skip_if_contains: Sequence[str] = ("norm",),
    select_layers=None,
) -> Tuple[torch.Tensor, List[dict], int]:
    """
    Same as collect_flat_weights but also returns metadata for reassembly.

    Returns:
      chunks:   (num_chunks, chunk_size)
      metadata: list of dicts per parameter group
      total_numel: total elements before padding (for unpadding)
    """
    if isinstance(skip_if_contains, str):
        skip_if_contains = (skip_if_contains,)
    if isinstance(select_layers, str):
        select_layers = (select_layers,)

    flat_parts = []
    metadata = []
    visited = set()
    offset = 0

    for name, w in state_dict.items():
        if name in visited:
            continue
        if any(s in name for s in skip_if_contains):
            continue
        if name.endswith(".bias"):
            continue
        if select_layers is not None and not any(st in name for st in select_layers):
            continue

        bias_name = name.replace(".weight", ".bias") if name.endswith(".weight") else None
        flat = w.detach().cpu().float().reshape(-1)
        visited.add(name)

        orig_shape = w.shape
        bias_shape = None
        w_numel = flat.numel()

        if bias_name and bias_name in state_dict:
            bias = state_dict[bias_name].detach().cpu().float().reshape(-1)
            bias_shape = state_dict[bias_name].shape
            flat = torch.cat([flat, bias])
            visited.add(bias_name)

        total_numel = flat.numel()

        metadata.append({
            "name": name,
            "bias_name": bias_name if bias_shape is not None else None,
            "orig_shape": orig_shape,
            "bias_shape": bias_shape,
            "w_numel": w_numel,
            "total_numel": total_numel,
            "offset": offset,
            "dtype": w.dtype,
        })

        offset += total_numel
        flat_parts.append(flat)

    if not flat_parts:
        return None, [], 0

    long_vec = torch.cat(flat_parts)
    total_len = long_vec.numel()

    n_chunks = math.ceil(total_len / chunk_size)
    padded_len = n_chunks * chunk_size
    pad_amount = padded_len - total_len
    if normalize == "min-max":
        long_vec = (long_vec - long_vec.min()) / (long_vec.max() - long_vec.min())
    elif normalize == "z_score":
        long_vec = (long_vec - long_vec.mean()) / long_vec.std()
    if pad_amount > 0:
        long_vec = F.pad(long_vec, (0, pad_amount), "constant", 0)

    chunks = long_vec.reshape(n_chunks, chunk_size)
    return chunks, metadata, total_len


def reassemble_state_dict(
    reconstructed_chunks: torch.Tensor,
    metadata: List[dict],
    total_numel: int,
    original_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Reverse of collect_flat_weights_with_metadata.

    Unchunk → remove padding → slice each param by offset → reshape.
    Params not in metadata are copied from original_state_dict unchanged.
    """
    new_sd = OrderedDict()

    for k, v in original_state_dict.items():
        new_sd[k] = v.clone()

    long_vec = reconstructed_chunks.reshape(-1)[:total_numel]

    for meta in metadata:
        start = meta["offset"]
        end = start + meta["total_numel"]
        flat = long_vec[start:end]
        orig_dtype = meta["dtype"]

        if meta["bias_name"] is not None:
            new_sd[meta["name"]] = flat[: meta["w_numel"]].reshape(meta["orig_shape"]).to(orig_dtype)
            new_sd[meta["bias_name"]] = flat[meta["w_numel"]:].reshape(meta["bias_shape"]).to(orig_dtype)
        else:
            new_sd[meta["name"]] = flat.reshape(meta["orig_shape"]).to(orig_dtype)

    return new_sd


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ZooDataset(Dataset):
    def __init__(self, datapath, dataset="Llama-3.2-3B-Instruct", split='train', topk=None,
                 scale=1.0, transform=None, normalize=False, tgt=None, exd=None, to_image=False,
                 in_ch=3, length=3072, n_tok=1, input_size=224, lamda=0.1):
        super().__init__()
        self.topk = topk
        self.split = split
        self.dataset = dataset
        self.length = length
        self.tgt = tgt
        self.exd = exd
        self.scale = scale
        self.n_tok = n_tok
        self.input_size = input_size
        self.to_image = to_image
        self.in_ch = in_ch
        self.lamda = lamda

        self.transform = transform
        self.normalize = normalize

        self.datapath = datapath
        data, mask = self._load_data(datapath, dataset=dataset)
        self.x_min = data.min()
        self.x_max = data.max()
        self.mu = data.mean()
        self.std = data.std()
        print(f'dataset size={data.shape}  max={data.max()}  min={data.min()}'
              f'--std={data.std()}--mean={data.mean()}  '
              f'pad kept={(mask == 0).sum().item()} elems')
        self.data = data.detach().cpu()
        self.mask = mask.detach().cpu()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        weight = self.data[idx].float().reshape(-1)
        mask = self.mask[idx].float().reshape(-1)

        if self.transform is not None:
            weight = self.transform(weight)
        weight = weight / self.scale
        return {'weight': weight, 'mask': mask, 'dataset': []}

    def _load_data(self, file, dataset="Llama-3.2-3B-Instruct"):
        models_list = load_models(file, dataset)
        print(f"Loading {len(models_list)} models")

        data, masks = [], []
        for model_path in models_list:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="cpu", torch_dtype=torch.bfloat16, trust_remote_code=True)
            std = model.state_dict()
            del model

            # ── Single-vector chunking (+ padding mask) ───────────
            wl, mask = collect_flat_weights(
                std,
                chunk_size=self.length * self.n_tok,
                normalize=self.normalize,
                skip_if_contains=self.exd,
                select_layers=self.tgt,
                return_mask=True,
            )

            if self.n_tok > 1:
                wl = wl.reshape(-1, self.n_tok, self.length)
                mask = mask.reshape(-1, self.n_tok, self.length)
            else:
                wl = wl.reshape(-1, self.length)
                mask = mask.reshape(-1, self.length)

            if self.to_image:
                wl = wl.reshape(-1, self.in_ch, self.input_size, self.input_size)
                mask = mask.reshape(-1, self.in_ch, self.input_size, self.input_size)

            data.append(wl)
            masks.append(mask)

        data = torch.cat(data, dim=0)
        mask = torch.cat(masks, dim=0)
        if self.topk is not None:
            data = data[:self.topk]
            mask = mask[:self.topk]
        return data, mask