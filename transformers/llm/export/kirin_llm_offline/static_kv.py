"""Experimental Qwen3-VL static graphs with explicit, host-managed KV storage.

The cache has a fixed capacity. Only its contents and the additive attention
mask change between calls. Every graph returns delta KV, never an enlarged
cache. This module does not use or mutate HF's internal DynamicCache.
"""
import numpy as np
import torch


KV_DTYPES = {"fp32": np.float32, "fp16": np.float16, "int8": np.int8}


def stable_rms_norm(hidden, norm):
    """RMSNorm whose intermediates remain safe when lowered to native FP16.

    FLOAT32 graph ports do not force all Kirin intermediates to FLOAT32.
    Scale before squaring and divide epsilon twice, so neither the residual
    square nor the scale square can overflow. The formula is unchanged.
    """
    if isinstance(norm, torch.nn.Identity):
        return hidden
    scale = hidden.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    unit = hidden / scale
    variance = unit.square().mean(dim=-1, keepdim=True)
    epsilon = norm.variance_epsilon / scale / scale
    return (unit * torch.rsqrt(variance + epsilon) * norm.weight).to(hidden.dtype)


def stable_attention_scores(query, keys, head_dim):
    # Avoid writing the larger unscaled QK dot product to an FP16 intermediate.
    factor = head_dim ** -0.25
    return torch.matmul(query * factor, (keys * factor).transpose(-1, -2))


def quantize_kv(value, group_size=32):
    """Symmetric INT8 per (batch, KV head, token, channel group), FP32 scales.

    Quantize only new, real tokens, after RoPE for K. Existing rows and their
    scales are immutable. All-zero rows use scale=1; -128 is never emitted.
    """
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 4 or not np.isfinite(value).all():
        raise ValueError("Expected finite rank-4 KV values")
    group_size = min(group_size, value.shape[-1])
    if group_size < 1 or value.shape[-1] % group_size:
        raise ValueError("KV group size must divide head_dim")
    grouped = value.reshape(*value.shape[:-1], -1, group_size)
    maximum = np.max(np.abs(grouped), axis=-1)
    scale = np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny)
    scale = np.where(maximum == 0, np.float32(1), scale)
    encoded = np.clip(np.rint(grouped / scale[..., None]), -127, 127).astype(np.int8).reshape(value.shape)
    return encoded, scale


def cache_memory(layers, kv_heads, capacity, head_dim, kv_dtype, group_size=32,
                 residual_tokens=128, sink_tokens=4):
    if kv_dtype not in KV_DTYPES:
        raise ValueError("Unknown KV dtype: " + kv_dtype)
    rows = 2 * layers * kv_heads * capacity
    data = rows * head_dim * np.dtype(KV_DTYPES[kv_dtype]).itemsize
    group_size = min(group_size, head_dim)
    if group_size < 1 or head_dim % group_size:
        raise ValueError("KV group size must divide head_dim")
    scales = rows * (head_dim // group_size) * 4 if kv_dtype == "int8" else 0
    recent = 2 * layers * kv_heads * min(residual_tokens, capacity) * head_dim * 2 if kv_dtype == "int8" else 0
    sinks = 2 * layers * kv_heads * min(sink_tokens, capacity) * head_dim * 2 if kv_dtype == "int8" else 0
    return {"dtype": kv_dtype, "data_bytes": data, "scale_bytes": scales,
            "residual_bytes": recent, "sink_bytes": sinks, "total_bytes": data + scales + recent + sinks}


class StaticDecoderLayer(torch.nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        if config.model_type != "qwen3_vl_text":
            raise ValueError("This prototype supports Qwen3-VL text layers only")
        self.layer = layer
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        if self.heads % self.kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")

    def forward(self, hidden, cos, sin, mask, past_key, past_value, deepstack):
        batch, tokens, _ = hidden.shape
        attention = self.layer.self_attn
        normalized = stable_rms_norm(hidden, self.layer.input_layernorm)
        query = stable_rms_norm(attention.q_proj(normalized).reshape(
            batch, tokens, self.heads, self.head_dim), attention.q_norm).transpose(1, 2)
        key = stable_rms_norm(attention.k_proj(normalized).reshape(
            batch, tokens, self.kv_heads, self.head_dim), attention.k_norm).transpose(1, 2)
        value = attention.v_proj(normalized).reshape(
            batch, tokens, self.kv_heads, self.head_dim).transpose(1, 2)

        def rotate_half(tensor):
            left, right = tensor.chunk(2, dim=-1)
            return torch.cat((-right, left), dim=-1)

        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        keys = torch.cat((past_key, key), dim=-2)
        values = torch.cat((past_value, value), dim=-2)
        groups = self.heads // self.kv_heads
        # Fold GQA query groups into the query dimension. Unlike repeat_kv,
        # this needs no rank-5 Expand/Tile and does not duplicate the KV cache.
        query = query.reshape(batch, self.kv_heads, groups * tokens, self.head_dim)
        scores = stable_attention_scores(query, keys, self.head_dim)
        scores = scores + mask.repeat(1, 1, groups, 1)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        probability = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        attended = torch.matmul(probability, values)
        attended = attended.reshape(batch, self.heads, tokens, self.head_dim)
        attended = attended.transpose(1, 2).reshape(batch, tokens, self.heads * self.head_dim)
        hidden = hidden + attention.o_proj(attended)
        hidden = hidden + self.layer.mlp(stable_rms_norm(hidden, self.layer.post_attention_layernorm))
        return hidden + deepstack, key, value


class StaticDecoderChunk(torch.nn.Module):
    def __init__(self, layers, config, kv_dtype="fp32", kv_group_size=32,
                 kv_residual_tokens=128, kv_sink_tokens=4):
        super().__init__()
        if kv_dtype not in KV_DTYPES:
            raise ValueError("Unknown KV dtype: " + kv_dtype)
        self.kv_dtype = kv_dtype
        self.kv_group_size = min(kv_group_size, config.head_dim)
        if self.kv_group_size < 1 or config.head_dim % self.kv_group_size:
            raise ValueError("KV group size must divide head_dim")
        self.residual_tokens = kv_residual_tokens if kv_dtype == "int8" else 0
        self.sink_tokens = kv_sink_tokens if kv_dtype == "int8" else 0
        self.layers = torch.nn.ModuleList([StaticDecoderLayer(layer, config) for layer in layers])

    def forward(self, hidden, cos, sin, mask, deepstack, *past_kv):
        deltas = []
        for i, layer in enumerate(self.layers):
            stride = (4 + 2 * bool(self.sink_tokens) + 2 * bool(self.residual_tokens)) if self.kv_dtype == "int8" else 2
            key, value = past_kv[stride * i:stride * i + 2]
            # Dequantization is inside the exported graph. Its boundary keeps
            # the cache compressed, rather than uploading an FP32 copy.
            if self.kv_dtype == "int8":
                key_scale, value_scale = past_kv[stride * i + 2:stride * i + 4]
                def dequantize(value, scale):
                    shape = value.shape
                    # Flatten groups to rank 2, avoiding a rank-5 Expand on
                    # mobile NPU. Every original group has its own scalar.
                    return (value.to(hidden.dtype).reshape(-1, self.kv_group_size)
                            * scale.reshape(-1, 1)).reshape(shape)
                key = dequantize(key, key_scale)
                value = dequantize(value, value_scale)
                offset = stride * i + 4
                for enabled in (self.sink_tokens, self.residual_tokens):
                    if enabled:
                        extra_key, extra_value = past_kv[offset:offset + 2]
                        key = torch.cat((key, extra_key.to(hidden.dtype)), dim=-2)
                        value = torch.cat((value, extra_value.to(hidden.dtype)), dim=-2)
                        offset += 2
            elif self.kv_dtype == "fp16":
                key, value = key.to(hidden.dtype), value.to(hidden.dtype)
            hidden, key, value = layer(hidden, cos, sin, mask,
                                       key, value, deepstack[i])
            # Return the small floating delta. The host encodes it once at
            # commit, leaving current-block attention in floating point.
            deltas.extend((key, value))
        return (hidden, *deltas)


class StaticHead(torch.nn.Module):
    def __init__(self, norm, projection):
        super().__init__()
        self.norm = norm
        self.projection = projection

    def forward(self, hidden):
        return self.projection(stable_rms_norm(hidden, self.norm))


class StaticVisionPre(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        projection = visual.patch_embed.proj
        self.weight = torch.nn.Parameter(projection.weight.detach().flatten(1), requires_grad=False)
        self.bias = projection.bias
        self.position = visual.pos_embed

    def forward(self, patches, indices, weights):
        # A non-overlapping Conv3D patch embedding is exactly this Linear.
        # It avoids rank-5 tensors in the compiled graph.
        patches = torch.nn.functional.linear(patches, self.weight, self.bias)
        positions = (self.position(indices) * weights.unsqueeze(-1)).sum(dim=1)
        return (patches + positions).unsqueeze(0)


class StaticVisionChunk(torch.nn.Module):
    def __init__(self, layers, start, deepstack_indexes):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.start = start
        self.deepstack_indexes = list(deepstack_indexes)

    def forward(self, hidden, cos, sin, mask):
        deepstack = []
        for index, layer in enumerate(self.layers):
            attention = layer.attn
            batch, tokens, width = hidden.shape
            projected = attention.qkv(layer.norm1(hidden))
            query, key, value = projected.chunk(3, dim=-1)
            query, key, value = [tensor.reshape(batch, tokens, attention.num_heads, attention.head_dim)
                                .transpose(1, 2) for tensor in (query, key, value)]
            def rotated(tensor):
                left, right = tensor.chunk(2, dim=-1)
                return tensor * cos + torch.cat((-right, left), dim=-1) * sin
            query, key = rotated(query), rotated(key)
            scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scaling + mask
            probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            output = torch.matmul(probabilities, value).transpose(1, 2).reshape(batch, tokens, width)
            hidden = hidden + attention.proj(output)
            hidden = hidden + layer.mlp(layer.norm2(hidden))
            if self.start + index in self.deepstack_indexes:
                deepstack.append(hidden)
        return (hidden, *deepstack)


class StaticVisionPost(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.merger = visual.merger
        self.deepstack = visual.deepstack_merger_list

    def forward(self, hidden, *deepstack):
        image = self.merger(hidden).unsqueeze(0)
        features = torch.stack([merger(value).unsqueeze(0)
                                for merger, value in zip(self.deepstack, deepstack)])
        return image, features


class StaticVisionFull(torch.nn.Module):
    """One visual graph, preserving post-block DeepStack order and mergers."""
    def __init__(self, visual):
        super().__init__()
        indexes = list(visual.deepstack_visual_indexes)
        if (not indexes or indexes != sorted(set(indexes)) or indexes[0] < 0
                or indexes[-1] >= len(visual.blocks)
                or len(indexes) != len(visual.deepstack_merger_list)):
            raise ValueError("Invalid visual DeepStack layer/merger mapping")
        self.pre = StaticVisionPre(visual)
        self.blocks = StaticVisionChunk(visual.blocks, 0, indexes)
        self.post = StaticVisionPost(visual)

    def forward(self, patches, indices, weights, cos, sin, mask):
        hidden = self.pre(patches, indices, weights)
        outputs = self.blocks(hidden, cos, sin, mask)
        return self.post(*outputs)


def attention_mask(capacity, past_length, real_tokens, block_tokens, dtype=np.float32,
                   residual_tokens=0, sink_tokens=0):
    if not 0 <= past_length or not 0 < real_tokens <= block_tokens or past_length + real_tokens > capacity:
        raise ValueError("KV capacity exceeded or invalid token lengths")
    # Finite FP16-safe masking. Padded query rows still see valid keys, so
    # Softmax never receives an all-negative-infinity row.
    if min(residual_tokens, sink_tokens) < 0:
        raise ValueError("Negative cache protection length")
    extra = residual_tokens + sink_tokens
    mask = np.full((1, 1, block_tokens, capacity + extra + block_tokens), -10000, dtype=dtype)
    # Each historical token appears exactly once: protected prefix, INT8
    # middle, or the FP16 ring. Overlapping copies remain masked.
    mask[..., sink_tokens:max(sink_tokens, past_length - residual_tokens)] = 0
    mask[..., capacity:capacity + min(sink_tokens, past_length)] = 0
    if residual_tokens:
        positions = np.arange(max(sink_tokens, past_length - residual_tokens), past_length)
        mask[..., capacity + sink_tokens + positions % residual_tokens] = 0
    elif not sink_tokens:
        mask[..., :past_length] = 0
    new_start = capacity + extra
    causal = np.arange(real_tokens)[None, :] <= np.arange(block_tokens)[:, None]
    mask[0, 0, :, new_start:new_start + real_tokens] = np.where(causal, 0, -10000)
    return mask


class KVCache:
    def __init__(self, layers, kv_heads, capacity, head_dim, dtype=np.float32, group_size=32,
                 residual_tokens=128, sink_tokens=4):
        if min(layers, kv_heads, capacity, head_dim) <= 0:
            raise ValueError("Cache dimensions must be positive")
        self.capacity = capacity
        self.length = 0
        if np.dtype(dtype) not in [np.dtype(value) for value in KV_DTYPES.values()]:
            raise ValueError("KV storage supports only FP32, FP16 or INT8")
        self.quantized = np.dtype(dtype) == np.dtype(np.int8)
        self.group_size = min(group_size, head_dim)
        if self.group_size < 1 or head_dim % self.group_size:
            raise ValueError("KV group size must divide head_dim")
        if min(residual_tokens, sink_tokens) < 0:
            raise ValueError("Negative cache protection length")
        self.residual_tokens = min(residual_tokens, capacity) if self.quantized else 0
        self.sink_tokens = min(sink_tokens, capacity) if self.quantized else 0
        self.key = [np.zeros((1, kv_heads, capacity, head_dim), dtype=dtype) for _ in range(layers)]
        self.value = [np.zeros_like(tensor) for tensor in self.key]
        shape = (1, kv_heads, capacity, head_dim // self.group_size)
        self.key_scale = [np.ones(shape, dtype=np.float32) for _ in range(layers)] if self.quantized else []
        self.value_scale = [np.ones(shape, dtype=np.float32) for _ in range(layers)] if self.quantized else []
        for kind, count in (("sink", self.sink_tokens), ("recent", self.residual_tokens)):
            shape = (1, kv_heads, count, head_dim)
            for field in ("key", "value"):
                setattr(self, kind + "_" + field,
                        [np.zeros(shape, dtype=np.float16) for _ in range(layers)] if count else [])

    def protected(self):
        return self.sink_key + self.sink_value + self.recent_key + self.recent_value

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self.key + self.value + self.key_scale + self.value_scale + self.protected())

    def inputs(self, start_layer, count):
        """Per layer: K, V, optional scales, sinks and recent KV; no decoding."""
        tensors = []
        for i in range(start_layer, start_layer + count):
            tensors.extend((self.key[i], self.value[i]))
            if self.quantized:
                tensors.extend((self.key_scale[i], self.value_scale[i]))
                if self.sink_tokens:
                    tensors.extend((self.sink_key[i], self.sink_value[i]))
                if self.residual_tokens:
                    tensors.extend((self.recent_key[i], self.recent_value[i]))
        return tensors

    def dequantized(self, layer):
        """Decode only the valid prefix for diagnostics; never used as graph input."""
        values = [self.key[layer][:, :, :self.length], self.value[layer][:, :, :self.length]]
        if self.quantized:
            scales = [self.key_scale[layer][:, :, :self.length], self.value_scale[layer][:, :, :self.length]]
            decoded = [(value.astype(np.float32).reshape(-1, self.group_size) * scale.reshape(-1, 1))
                       .reshape(value.shape) for value, scale in zip(values, scales)]
            for field, value in zip(("key", "value"), decoded):
                prefix = min(self.length, self.sink_tokens)
                if prefix:
                    value[:, :, :prefix] = getattr(self, "sink_" + field)[layer][:, :, :prefix]
                if self.residual_tokens:
                    positions = np.arange(max(self.sink_tokens, self.length - self.residual_tokens), self.length)
                    value[:, :, positions] = getattr(self, "recent_" + field)[layer][:, :, positions % self.residual_tokens]
            return tuple(decoded)
        return tuple(value.astype(np.float32, copy=False) for value in values)

    def reset(self):
        self.length = 0
        for tensor in self.key + self.value + self.protected():
            tensor.fill(0)
        for scale in self.key_scale + self.value_scale:
            scale.fill(1)

    def append_chunk(self, start_layer, outputs, real_tokens):
        end = self.length + real_tokens
        count = len(outputs) // 2
        if (end > self.capacity or real_tokens < 1 or len(outputs) % 2
                or start_layer < 0 or start_layer + count > len(self.key)):
            raise ValueError("Invalid KV append")
        prepared = []
        for local in range(count):
            index = start_layer + local
            for kind, storage, delta in (("key", self.key[index], outputs[2 * local]),
                                         ("value", self.value[index], outputs[2 * local + 1])):
                expected = (1, storage.shape[1], real_tokens, storage.shape[3])
                if delta.ndim != 4 or delta.shape[2] < real_tokens or delta[:, :, :real_tokens].shape != expected:
                    raise ValueError("Invalid delta KV shape")
                values = delta[:, :, :real_tokens]
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite delta KV")
                protected = None
                if self.quantized and (self.sink_tokens or self.residual_tokens):
                    with np.errstate(over="ignore"):
                        protected = values.astype(np.float16)
                    if not np.isfinite(protected).all():
                        raise ValueError("Delta KV overflows protected FP16 cache")
                if self.quantized:
                    values, scale = quantize_kv(values, self.group_size)
                else:
                    with np.errstate(over="ignore"):
                        values = values.astype(storage.dtype, copy=False)
                    if not np.isfinite(values).all():
                        raise ValueError("Delta KV overflows cache dtype")
                    scale = None
                prepared.append((kind, index, values, scale, protected))
        # Validate/encode the complete chunk before writing any of its slots.
        for kind, index, values, scale, protected in prepared:
            getattr(self, kind)[index][:, :, self.length:end] = values
            if scale is not None:
                getattr(self, kind + "_scale")[index][:, :, self.length:end] = scale
            if protected is not None:
                prefix = min(real_tokens, max(0, self.sink_tokens - self.length))
                if prefix:
                    getattr(self, "sink_" + kind)[index][:, :, self.length:self.length + prefix] = protected[:, :, :prefix]
                if self.residual_tokens:
                    offsets = np.arange(max(0, real_tokens - self.residual_tokens), real_tokens)
                    slots = (self.length + offsets) % self.residual_tokens
                    getattr(self, "recent_" + kind)[index][:, :, slots] = protected[:, :, offsets]

    def commit(self, count):
        if count < 1 or self.length + count > self.capacity:
            raise ValueError("KV capacity exceeded")
        self.length += count


class StaticSession:
    """Host reference scheduler; callbacks can run Torch, ORT, or a device graph.

    Each callback receives (hidden, cos, sin, mask, deepstack, K0, V0, ...)
    and returns (hidden, delta_K0, delta_V0, ...). Chunks run synchronously;
    global cache length advances only after all layers finish one token block.
    The session accepts already embedded tokens and precomputed M-RoPE.
    """
    def __init__(self, chunks, config, capacity, prefill_tokens, dtype=np.float32, kv_dtype="fp32", kv_group_size=32,
                 kv_residual_tokens=128, kv_sink_tokens=4):
        if prefill_tokens < 1 or prefill_tokens > capacity:
            raise ValueError("Invalid prefill block size")
        self.chunks = chunks  # (global_start, layer_count, callback)
        coverage = [i for start, count, _ in chunks for i in range(start, start + count)]
        if coverage != list(range(config.num_hidden_layers)):
            raise ValueError("Chunks must cover all decoder layers exactly once in order")
        self.config = config
        self.prefill_tokens = prefill_tokens
        self.dtype = dtype
        self.kv_dtype = kv_dtype
        if kv_dtype not in KV_DTYPES:
            raise ValueError("Unknown KV dtype: " + kv_dtype)
        self.cache = KVCache(config.num_hidden_layers, config.num_key_value_heads,
                             capacity, config.head_dim, KV_DTYPES[kv_dtype], kv_group_size,
                             kv_residual_tokens, kv_sink_tokens)

    def forward(self, embeddings, cos, sin, deepstack=None, decode=False):
        original_length = self.cache.length
        # Ring writes can overwrite valid old slots before commit. Preserve
        # only this small FP16 window for rollback, never the full cache.
        original_recent = ([value.copy() for value in self.cache.recent_key + self.cache.recent_value]
                           if original_length else [])
        try:
            return self._forward(embeddings, cos, sin, deepstack, decode)
        except Exception:
            # Slots written after the original prefix remain masked on retry.
            # This also rolls back earlier blocks in a multi-block prefill.
            self.cache.length = original_length
            for target, saved in zip(self.cache.recent_key + self.cache.recent_value, original_recent):
                np.copyto(target, saved)
            raise

    def _forward(self, embeddings, cos, sin, deepstack, decode):
        tokens = embeddings.shape[1]
        if embeddings.shape != (1, tokens, self.config.hidden_size) or tokens < 1:
            raise ValueError("Expected nonempty [1, tokens, hidden_size] embeddings")
        if cos.shape != (1, 1, tokens, self.config.head_dim) or sin.shape != cos.shape:
            raise ValueError("Invalid M-RoPE shape")
        if decode and tokens != 1:
            raise ValueError("Decode graph accepts one token")
        if self.cache.length + tokens > self.cache.capacity:
            raise ValueError("KV capacity exceeded; reset or rebuild with a larger context")
        if deepstack is not None and (deepstack.ndim != 4 or deepstack.shape[1:] != embeddings.shape
                                      or deepstack.shape[0] > self.config.num_hidden_layers):
            raise ValueError("Invalid dense DeepStack features")
        width = 1 if decode else self.prefill_tokens
        results = []
        for offset in range(0, tokens, width):
            real = min(width, tokens - offset)
            hidden = np.zeros((1, width, self.config.hidden_size), dtype=self.dtype)
            hidden[:, :real] = embeddings[:, offset:offset + real]
            block_cos = np.ones((1, 1, width, self.config.head_dim), dtype=self.dtype)
            block_sin = np.zeros_like(block_cos)
            block_cos[:, :, :real] = cos[:, :, offset:offset + real]
            block_sin[:, :, :real] = sin[:, :, offset:offset + real]
            mask = attention_mask(self.cache.capacity, self.cache.length, real, width, self.dtype,
                                  self.cache.residual_tokens, self.cache.sink_tokens)
            # A failed callback may have written uncommitted slots. They stay
            # masked, and a retry overwrites them before committing any length.
            for start, count, callback in self.chunks:
                features = np.zeros((count, 1, width, self.config.hidden_size), dtype=self.dtype)
                if deepstack is not None:
                    for local in range(count):
                        if start + local < len(deepstack):
                            features[local, :, :real] = deepstack[start + local, :, offset:offset + real]
                past = self.cache.inputs(start, count)
                outputs = callback(decode, [hidden, block_cos, block_sin, mask, features, *past])
                if len(outputs) != 1 + 2 * count or outputs[0].shape != hidden.shape:
                    raise ValueError("Invalid decoder chunk outputs")
                if any(not np.isfinite(tensor).all() for tensor in outputs):
                    raise ValueError("Decoder produced non-finite values")
                self.cache.append_chunk(start, outputs[1:], real)
                hidden = outputs[0]
            self.cache.commit(real)
            results.append(hidden[:, :real].copy())
        return np.concatenate(results, axis=1)
