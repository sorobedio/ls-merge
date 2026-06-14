import argparse
import os
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import yaml


def load_yaml_as_dict(yaml_content):
    """
    Load a YAML content as a dictionary.

    Parameters:
        yaml_content (str): The YAML content in string format.

    Returns:
        dict: A dictionary containing the parsed YAML structure.
    """
    try:
        # Use yaml.safe_load to parse the YAML content
        yaml_dict = yaml.safe_load(yaml_content)
        return yaml_dict
    except yaml.YAMLError as exc:
        print(f"Error loading YAML: {exc}")
        return None



def get_basic_blk_weights(model, nbk=7, model_name=None):
    weights = {}

    # Loop through the number of transformer blocks (nbk)
    for i in range(nbk):
        # Define the layer path
        layer_path = f'model.encoder.layers.encoder_layer_{i}'

        # If model_name is provided, append it to the key
        if model_name is not None:
            key_prefix = f'{model_name}-model.encoder.layers.encoder_layer_{i}'
        else:
            key_prefix = f'model.encoder.layers.encoder_layer_{i}'

        # Access the transformer block by evaluating its path dynamically
        block_layer = eval(layer_path)

        # Get the state_dict of the transformer block
        std = block_layer.state_dict()

        # Assuming `gets_weights` is a function that processes the state_dict
        w = gets_weights(std)
        print(f'-------------{w.shape}-------------')

        # Store the weights in the dictionary with the prefixed key
        weights[key_prefix] = w

    return weights


def add_to_config(mydict, cfl):
    with open(cfl, 'a') as configfile:
        data = yaml.dump(mydict, configfile, indent=4)
        print("Write successful")

def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def set_state_dict(std, weights):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            device = std[params].device
            dtp = std[params].dtype
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
            # model.load_state_dict(std)
            st = ed
    return std


def set_norm_state_dict(std, weights, tg='norm'):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if tg in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + np.prod(shape)
                std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
    return std


def get_layer_weights(std, tgt='norm'):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                w = std[params].reshape(-1)
                weights.append(w)
    return torch.cat(weights, -1)

def set_layer_state_dict(std, weights, layer='mlp'):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if layer in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + int(np.prod(shape))
                # print(ed)

                std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
                # print(f'setting parameters---{params}')
    return std


def sset_layer_state_dict(std, weights, layer='mlp'):
    # std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if layer in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + int(np.prod(shape))
                # print(ed)

                std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
                # print(f'setting parameters---{params}')
    return std


def set_layer_state_dict_vect(state_dict: dict,
                              flat_weights_dict: dict) -> dict:
    """
    state_dict:   the model.state_dict() you want to overwrite
    flat_weights_dict:  { layer_key: 1D tensor of concatenated weights }

    For each layer_key, finds all param names in state_dict containing that key,
    then slices flat_weights_dict[layer_key] into chunks matching each param’s shape.
    """
    # Make a copy so we don’t clobber the original dict
    new_sd = state_dict.copy()

    for layer_key, flat_w in flat_weights_dict.items():
        offset = 0
        # collect all params whose name contains this layer_key
        matches = [n for n in new_sd if layer_key in n]
        if not matches:
            raise KeyError(f"No parameters found matching '{layer_key}'")

        for name in matches:
            target = new_sd[name]
            numel = target.numel()
            end = offset + numel

            # slice + reshape + cast + move back to original device
            chunk = flat_w[offset:end]
            if chunk.numel() != numel:
                raise ValueError(
                    f"Layer '{layer_key}', param '{name}': "
                    f"expected {numel} elements but got {chunk.numel()}"
                )

            new_sd[name] = (chunk
                            .reshape(target.shape)
                            .to(dtype=target.dtype,
                                device=target.device))
            offset = end

        # sanity check
        total = flat_w.numel()
        if offset != total:
            raise ValueError(
                f"Layer '{layer_key}': consumed {offset}/{total} elements"
            )

        print(f"✅ set parameters for layer key '{layer_key}'")

    return new_sd




def set_layer_state_dict_vect(std, weights):
    # std = model.state_dict()

    layers = list(std)
    tlayers = list(weights)
    for p in tlayers:
        st = 0
        weight = weights[p]
        for params in layers:
            if p in  params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                ed = st + int(np.prod(shape))
                # print(ed)
                std[params] = weight[st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
        print(f'setting parameters---{p}')

    return std

def extract_layer_weights_with_b(std, tgt='norm'):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1,-1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)]=w
    return weights, ws


def extract_layers_with_b(std, tgt='norm'):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt not in params:
                if 'bias' in params:
                    continue
                w = std[params].reshape(1, -1)
                p = params.replace('weight', 'bias')
                try:
                    b = std[p].reshape(1, -1)
                    print(f'paramertes============={params}-------{w.shape}------{b.shape}--------')
                    w = torch.cat((w, b), dim=-1)
                except:
                    print(f'paramertes============={params}-------{w.shape}------no bias--------')

                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                weights[str(params)] = w
    return weights, ws

def set_layers_state_dict_merge(std, weights, t=1.5):
    # std = model.state_dict()
    # layers = list(std)
    tlayers = list(weights)
    # st = 0

    for p in tlayers:
        # if p == params:
        shape = std[p].shape
        device = std[p].device
        dtp = std[p].dtype
        st =0
        ed = st + np.prod(shape)
        b =weights[p][st:ed].reshape(shape).type(dtp).to(device)
        std[p] = b + t *(std[p] - b)
        # std[p] = std[p] +t*b
        st = ed
        # model.load_state_dict(std)
        # st = ed
    return std

def set_layers_state_dict(std, weights):
    # std = model.state_dict()
    # layers = list(std)
    tlayers = list(weights)
    # st = 0

    for p in tlayers:
        # if p == params:
        shape = std[p].shape
        device = std[p].device
        dtp = std[p].dtype
        st =0
        ed = st + np.prod(shape)
        std[p] = weights[p][st:ed].reshape(shape).type(dtp).to(device)
        # model.load_state_dict(std)
        # st = ed
    return std


def set_mat_layers_state_dict(std, weights):
    # std = model.state_dict()
    layers = list(weights)
    # st = 0
    for params in layers:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            device = std[params].device
            dtp = std[params].dtype
            st =0
            ed = st + shape[1]
            std[params] = weights[params][:shape[0],st:ed].reshape(shape).type(dtp).to(device)
            # model.load_state_dict(std)
            # st = ed
    return std


def set_mat_with_b_weights(model, weights_vectorized):
    """
    Restore the model's MLP weights from a provided dictionary of vectorized weight tensors.

    Parameters:
    - model: A PyTorch model with MLP layers. The model must already be initialized with
      its architecture so we know the original shapes of the parameters.
    - mlp_weights_vectorized (dict): A dictionary mapping parameter names to flattened tensors.
      Some entries may include bias vectors concatenated immediately after the corresponding weight
      vector, and potentially some padding.

    Returns:
    - model: The model with restored MLP parameters.
    """
    if weights_vectorized is None:
        raise ValueError("Please provide a dictionary of MLP weights in vectorized form.")

    # Get the current state dictionary of the model
    state_dict = model.state_dict()

    # Iterate over the provided MLP weights
    for name, vector in weights_vectorized.items():
        # Check if the parameter name exists in the current model state
        if name not in state_dict:
            # If not found, skip or raise an error as needed
            # raise KeyError(f"Parameter '{name}' not found in model state.")
            continue

        # Determine the original shape of the parameter and how many elements it has
        original_shape = state_dict[name].shape
        weight_size = state_dict[name].numel()
        dt = state_dict[name].dtype

        # If the vector is longer than the original weight size,
        # we assume the extra elements correspond to the bias vector.
        if vector.numel() > weight_size:
            weight_vector = vector[:weight_size].to(dt)
            bias_vector = vector[weight_size:].to(dt)

            # Reshape and copy back the weight parameter
            state_dict[name].copy_(weight_vector.view(original_shape))

            # Identify the bias parameter name by replacing "weight" with "bias"
            bias_name = name.replace("weight", "bias")

            # Ensure the bias exists
            if bias_name in state_dict:
                bias_original_shape = state_dict[bias_name].shape
                bias_size = state_dict[bias_name].numel()

                # If the provided vector has more elements than needed for bias
                # (e.g., due to padding), we slice only the required portion.
                bias_vector = bias_vector[:bias_size].to(dt)
                state_dict[bias_name].copy_(bias_vector.view(bias_original_shape))
            else:
                # If there's no corresponding bias in the model, this might be unexpected.
                # You can raise an error or log a warning here.
                pass
        else:
            # If no bias is present, just reshape and set the weight tensor
            state_dict[name].copy_(vector.view(original_shape))

    # Load the modified state dictionary back into the model
    model.load_state_dict(state_dict)

    return model



def set_layers_state_dict_ecp(std, weights, cond='norm', tgt='mlp'):
    # std = model.state_dict()
    layers = list(weights)
    # st = 0
    for params in layers:
        if not params.endswith('num_batches_tracked'):
            if cond in params:
                continue
            if tgt in params:
                shape = std[params].shape
                device = std[params].device
                dtp = std[params].dtype
                st =0
                ed = st + np.prod(shape)
                std[params] = weights[params][st:ed].reshape(shape).type(dtp).to(device)
                # model.load_state_dict(std)
                st = ed
    return std


def set_lora_ab_weights(sd: dict,
                        packed_weights: dict,
                        ):
    # sd = model.state_dict()
    for packed_key, flat in packed_weights.items():
        # remove model_name prefix
        # assert packed_key.startswith(model_name + "__")
        # orig_key = packed_key[len(model_name + "__"):]
        orig_key = packed_key.split("__")[-1]
        # derive the two target keys
        key_A = orig_key
        key_B = orig_key.replace("lora_A", "lora_B")
        A_shape = sd[key_A].shape
        B_shape = sd[key_B].shape
        # split the flat vector back into A and B parts
        total_dim = flat.shape[1]
        A_numel = A_shape[0] * A_shape[1]
        B_numel = B_shape[0] * B_shape[1]
        assert A_numel + B_numel <= total_dim, (
            f"Mismatch: A+B = {A_numel+B_numel} vs flat {total_dim}"
        )

        # slice and reshape
        flat = flat.reshape(-1)  # (A_numel + B_numel,)
        a_flat = flat[:A_numel]
        b_flat = flat[A_numel:]

        wA = a_flat.view(A_shape)
        wB = b_flat.view(B_shape)
        # write back into sd
        sd[key_A].copy_(wA)
        sd[key_B].copy_(wB)
        # print(f"Unpacked {packed_key} → {key_A}{A_shape}, {key_B}{B_shape}")
    # load back
    # model.load_state_dict(sd, strict=False)
    return sd


def gets_weights(std):
    # std = model.state_dict()
    weights = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            w = std[params].reshape(-1)
            weights.append(w)
    return torch.cat(weights, -1)


def set_model_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if params.endswith('running_var') or params.endswith('running_mean'):
                continue
            elif 'rotary_emb' in params:
                print(f'found------{params}-------------')
                continue
            # elif 'linear' in params:
            #     continue
            shape = std[params].shape
            dtp = std[params].dtype
            device = std[params].device
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape).type(dtp).to(device)
            model.load_state_dict(std)
            st = ed
    return model

def extract_layer_weights(std, tgt='norm', pref=None):
    # std = model.state_dict()
    weights = {}
    ws = []
    for params in std:
        if not params.endswith('num_batches_tracked'):
            if 'mean' in params or 'var' in params:
                continue
            # print(params)
            if tgt in params:
                w = std[params].reshape(1,-1)
                print(f'paramertes============={params}---------------------')
                print(w.shape)
                print(w.min(), w.max())
                ws.append(w)
                if pref is not None:
                    key = f'{pref}_{str(params)}'
                # weights[key]=w
                else:
                    key = f'{str(params)}'
                weights[key] = w
    ws = torch.cat(ws, dim=-1)
    return weights, ws

def set_weights(model, weights):
    std = model.state_dict()
    st = 0
    for params in std:
        if not params.endswith('num_batches_tracked'):
            shape = std[params].shape
            ed = st + np.prod(shape)
            std[params] = weights[st:ed].reshape(shape)
            model.load_state_dict(std)
            st = ed



def vecpadder(x, max_in=3728761 * 3):
    shape = x.shape
    delta1 = max_in - shape[0]
    x = F.pad(x, (0, delta1))
    return x


def pad_to_chunk_multiple(x, chunk_size):
    shape = x.shape
    if len(shape)<2:
        x =x.unsqueeze(0)
        shape = x.shape
    max_in = chunk_size*math.ceil(shape[1]/chunk_size)
    delta1 = max_in - shape[1]
    # x = F.pad(x, (0, delta1))
    x =F.pad(x, (0, delta1, 0, 0), "constant", 0)
    return x

def matpadder(x, max_in=512):
    shape =x.shape
    # delta1 = max_in - shape[0]
    delta2 = max_in - shape[1]

    out = F.pad(x, (0, delta2, 0, 0), "constant", 0)
    return out


def get_weights_mat(std,xcond, typ=None):
    weights ={}
    if typ is None:
        for k in std:
            if 'weight_scale' in k:
                continue
            if xcond is not None and xcond in k:
                continue
            w = std[k].detach().cpu()
            weights[k] = w.reshape(1, -1)
            print(f'param:{k}--shape:{w.shape}--min:{w.min()}--max:{w.max()}')
            print('-----------------------------------------------------------')

    else:
        for k in std:
            if 'weight_scale' in k:
                continue
            if xcond is not None and xcond in k:
                continue
            if typ in k:
                w = std[k].detach().cpu()
                weights[k] = w
                print(f'param:{k}--shape:{w.shape}--min:{w.min()}--max:{w.max()}')
                print('-----------------------------------------------------------')
    return weights



def set_weights_mat(std, weights, xcond=None, typ=None):
    """
    Reverse of get_weights_mat: inject tensors from `weights` back into `std`.

    Args:
        std (dict[str, Tensor]): original state_dict (key → Tensor).
        weights (dict[str, Tensor]): mapping from keys to CPU Tensors.
        xcond (str, optional): if provided, skip any key containing this substring.
        typ (str, optional): if provided, only set keys containing this substring.

    Returns:
        new_std (dict[str, Tensor]): a copy of std with entries replaced by weights.
    """
    new_std = {}
    st =0
    for k, v in std.items():

        # skip by xcond
        if xcond is not None and xcond in k:
            new_std[k] = v
            continue

        # if typ filter is set, only replace matching keys
        if typ is None or typ in k:
            if k in weights:
                shape = v.shape
                ed = st + np.prod(shape)
                w = weights[k][:,st:ed].reshape(shape)
                # cast back to original device and dtype
                new_std[k] = w.to(device=v.device, dtype=v.dtype)
                print(f'setting param:{k} from weights dict (shape {w.shape})')
                st = ed
            else:
                # no match in weights dict → leave original
                new_std[k] = v
        else:
            # typ filter excludes this key → leave original
            new_std[k] = v

    return new_std


def window_tensor(x, window_rows, window_cols, step_rows=None, step_cols=None):
    T, L = x.shape
    if step_rows is None:
        step_rows = window_rows  # non-overlapping by default
    if step_cols is None:
        step_cols = window_cols  # non-overlapping by default
    if T < window_rows or L < window_cols:
        raise ValueError("Tensor dimensions must be at least as large as the window size.")
    # Use unfold to extract sliding windows along both dimensions.
    windows = x.unfold(0, window_rows, step_rows).unfold(1, window_cols, step_cols)
    windows = windows.reshape(-1, window_rows, window_cols)
    return windows


def reconstruct_from_windows(windows, original_shape=None, step_rows=None, step_cols=None):
    """
    Reconstruct the original 2D tensor from windows.

    For non-overlapping windows:
      - Provide a 4D tensor with shape
        (num_window_rows, num_window_cols, window_rows, window_cols)
      - Do not pass original_shape or step sizes (or let them default to None).
      The function will simply permute and reshape the tensor.

    For overlapping windows:
      - Provide original_shape as a tuple (T, L) of the original tensor.
      - Optionally, specify step_rows and step_cols; if not provided they default to
        the window dimensions (non-overlapping).
      The function will accumulate contributions from overlapping regions
      and average them.

    Parameters:
        windows (torch.Tensor): 4D tensor with shape
                                (num_window_rows, num_window_cols, window_rows, window_cols).
        original_shape (tuple, optional): Shape (T, L) of the original tensor.
        step_rows (int, optional): Step size along rows. Defaults to window_rows.
        step_cols (int, optional): Step size along columns. Defaults to window_cols.

    Returns:
        torch.Tensor: The reconstructed 2D tensor.
    """
    num_window_rows, num_window_cols, window_rows, window_cols = windows.shape

    # If no original_shape and no steps provided, assume non-overlapping windows.
    if original_shape is None and step_rows is None and step_cols is None:
        # Non-overlapping: simply rearrange the windows back into a 2D tensor.
        x_reconstructed = windows.permute(0, 2, 1, 3).contiguous()
        x_reconstructed = x_reconstructed.view(num_window_rows * window_rows,
                                               num_window_cols * window_cols)
        return x_reconstructed
    else:
        # For overlapping windows, we require the original tensor shape.
        if original_shape is None:
            raise ValueError("For overlapping windows, original_shape must be provided.")
        T, L = original_shape
        if step_rows is None:
            step_rows = window_rows
        if step_cols is None:
            step_cols = window_cols

        # Prepare tensors for reconstruction and to count contributions.
        output = torch.zeros(original_shape, dtype=windows.dtype, device=windows.device)
        count = torch.zeros(original_shape, dtype=windows.dtype, device=windows.device)

        # Loop over each window and place it in the corresponding region of the output.
        for i in range(num_window_rows):
            for j in range(num_window_cols):
                start_i = i * step_rows
                start_j = j * step_cols
                # Add the window's values into the output tensor.
                output[start_i:start_i + window_rows, start_j:start_j + window_cols] += windows[i, j]
                # Keep track of how many times each element is covered.
                count[start_i:start_i + window_rows, start_j:start_j + window_cols] += 1

        # Avoid division by zero.
        count[count == 0] = 1
        # Average the overlapping contributions.
        return output / count


def reconstruct_from_windows(windows):
    """
    Reconstruct the original tensor from non-overlapping windows.

    Parameters:
        windows (torch.Tensor): 4D tensor with shape
                                (num_window_rows, num_window_cols, window_rows, window_cols).

    Returns:
        torch.Tensor: The reconstructed 2D tensor.
    """
    num_window_rows, num_window_cols, window_rows, window_cols = windows.shape
    # Permute dimensions so that window rows and columns come next to their group dimensions.
    x_reconstructed = windows.permute(0, 2, 1, 3).contiguous()
    # Reshape to combine the groups into the original 2D tensor.
    x_reconstructed = x_reconstructed.view(num_window_rows * window_rows, num_window_cols * window_cols)
    return x_reconstructed


def pad_to_token_and_length_multiple(x, chunk_size, tokens=0):
    shape = x.shape
    delta2 = 0
    delta1 = 0
    # If input is less than 2 dimensions, unsqueeze to add a batch dimension.
    if len(shape) < 2:
        x = x.unsqueeze(0)
        shape = x.shape

    # Calculate padding for the second dimension (columns)
    max_in = chunk_size * math.ceil(shape[1] / chunk_size)
    if max_in > shape[1]:
        delta1 = max_in - shape[1]

    # If tokens parameter is provided, calculate padding for the first dimension (rows)
    if tokens > 0:
        max_t = tokens * math.ceil(shape[0] / tokens)
        if max_t > shape[0]:
            delta2 = max_t - shape[0]

    # Apply zero padding: F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    x = F.pad(x, (0, delta1, 0, delta2), "constant", 0)
    return x