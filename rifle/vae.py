# Derived from Ultimate VAE Tile Optimization by @pkuliyi2015.
# Upstream declaration: MIT License; retained lineage notices are in licenses/.
# RIFLE modifications synchronize global GroupNorm and attention across tiles.
# ------------------------------------------------------------------------
#
#   Ultimate VAE Tile Optimization (modified)
#
#   Modified version:
#   1) exact global GroupNorm statistics using sum / sq_sum / count
#   2) fast GN estimation disabled internally for better color consistency
#   3) attention is synchronized globally across all tiles
#
# ------------------------------------------------------------------------

import gc
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from rifle import devices


def inplace_nonlinearity(x):
    return F.silu(x, inplace=True)


def attention_forward(self, h_):
    batch_size, channel, height, width = h_.shape
    hidden_states = h_.view(batch_size, channel, height * width).transpose(1, 2)

    attention_mask = None
    encoder_hidden_states = None
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = self.prepare_attention_mask(
        attention_mask, sequence_length, batch_size
    )

    query = self.to_q(hidden_states)

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)

    attention_probs = self.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    hidden_states = self.to_out[0](hidden_states)
    hidden_states = self.to_out[1](hidden_states)

    hidden_states = hidden_states.transpose(-1, -2).reshape(
        batch_size, channel, height, width
    )

    return hidden_states


def run_attention_layer(net, hidden):
    return attention_forward(net, hidden)


def attn2task(task_queue, net):
    task_queue.append(("store_res", lambda x: x))

    if hasattr(net, "group_norm"):
        task_queue.append(("pre_norm", net.group_norm))
    elif hasattr(net, "norm"):
        task_queue.append(("pre_norm", net.norm))
    else:
        raise AttributeError(
            "Unsupported attention block: cannot find group_norm / norm"
        )

    task_queue.append(("attn_sync", net))
    task_queue.append(["add_res", None])


def resblock2task(queue, block):
    if block.in_channels != block.out_channels:
        if block.use_in_shortcut:
            queue.append(("store_res", block.conv_shortcut))
        else:
            queue.append(("store_res", block.nin_shortcut))
    else:
        queue.append(("store_res", lambda x: x))

    queue.append(("pre_norm", block.norm1))
    queue.append(("silu", inplace_nonlinearity))
    queue.append(("conv1", block.conv1))
    queue.append(("pre_norm", block.norm2))
    queue.append(("silu", inplace_nonlinearity))
    queue.append(("conv2", block.conv2))
    queue.append(["add_res", None])


def build_sampling(task_queue, net, is_decoder):
    if is_decoder:
        resblock2task(task_queue, net.mid_block.resnets[0])
        attn2task(task_queue, net.mid_block.attentions[0])
        resblock2task(task_queue, net.mid_block.resnets[1])

        resolution_iter = range(len(net.up_blocks))
        block_ids = 2 + 1
        condition = len(net.up_blocks) - 1
        module = net.up_blocks
        func_name = "upsamplers"
    else:
        resolution_iter = range(len(net.down_blocks))
        block_ids = 2
        condition = len(net.down_blocks) - 1
        module = net.down_blocks
        func_name = "downsamplers"

    for i_level in resolution_iter:
        for i_block in range(block_ids):
            resblock2task(task_queue, module[i_level].resnets[i_block])
        if i_level != condition:
            if is_decoder:
                task_queue.append((func_name, module[i_level].upsamplers[0]))
            else:
                task_queue.append((func_name, module[i_level].downsamplers[0]))

    if not is_decoder:
        resblock2task(task_queue, net.mid_block.resnets[0])
        attn2task(task_queue, net.mid_block.attentions[0])
        resblock2task(task_queue, net.mid_block.resnets[1])


def build_task_queue(net, is_decoder):
    task_queue = []
    task_queue.append(("conv_in", net.conv_in))

    build_sampling(task_queue, net, is_decoder)

    task_queue.append(("pre_norm", net.conv_norm_out))
    task_queue.append(("silu", inplace_nonlinearity))
    task_queue.append(("conv_out", net.conv_out))

    return task_queue


def clone_task_queue(task_queue):
    return [[item for item in task] for task in task_queue]


def custom_group_norm(input, num_groups, mean, var, weight=None, bias=None, eps=1e-6):
    b, c = input.size(0), input.size(1)
    channel_in_group = int(c / num_groups)
    input_reshaped = input.contiguous().view(
        1, int(b * num_groups), channel_in_group, *input.size()[2:]
    )

    out = F.batch_norm(
        input_reshaped,
        mean,
        var,
        weight=None,
        bias=None,
        training=False,
        momentum=0,
        eps=eps,
    )

    out = out.view(b, c, *input.size()[2:])

    if weight is not None:
        out *= weight.view(1, -1, 1, 1)
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    return out


def crop_valid_region(x, input_bbox, target_bbox, is_decoder):
    padded_bbox = [i * 8 if is_decoder else i // 8 for i in input_bbox]
    margin = [target_bbox[i] - padded_bbox[i] for i in range(4)]
    return x[:, :, margin[2] : x.size(2) + margin[3], margin[0] : x.size(3) + margin[1]]


def scale_bbox(bbox, scale_num=1, scale_den=1):
    return [
        bbox[0] * scale_num // scale_den,
        bbox[1] * scale_num // scale_den,
        bbox[2] * scale_num // scale_den,
        bbox[3] * scale_num // scale_den,
    ]


def crop_valid_region_with_scale(x, input_bbox, target_bbox, scale_num=1, scale_den=1):
    scaled_input_bbox = scale_bbox(input_bbox, scale_num, scale_den)
    margin = [target_bbox[i] - scaled_input_bbox[i] for i in range(4)]
    return x[:, :, margin[2] : x.size(2) + margin[3], margin[0] : x.size(3) + margin[1]]


class GroupNormParam:
    def __init__(self):
        self.sum_list = []
        self.sq_sum_list = []
        self.count_list = []
        self.weight = None
        self.bias = None

    def add_tile(self, tile, layer, num_groups=32):
        x = tile.float()
        b, c = x.size(0), x.size(1)
        ch_per_group = c // num_groups

        x = x.contiguous().view(1, b * num_groups, ch_per_group, *x.size()[2:])

        s = x.sum(dim=[0, 2, 3, 4])
        sq = (x * x).sum(dim=[0, 2, 3, 4])
        cnt = ch_per_group * tile.shape[2] * tile.shape[3]

        self.sum_list.append(s)
        self.sq_sum_list.append(sq)
        self.count_list.append(cnt)

        if hasattr(layer, "weight"):
            self.weight = layer.weight.float() if layer.weight is not None else None
            self.bias = layer.bias.float() if layer.bias is not None else None
        else:
            self.weight = None
            self.bias = None

    def summary(self, num_groups=32):
        if len(self.sum_list) == 0:
            return None

        total_sum = torch.stack(self.sum_list, dim=0).sum(dim=0)
        total_sq_sum = torch.stack(self.sq_sum_list, dim=0).sum(dim=0)
        total_count = float(sum(self.count_list))

        mean = total_sum / total_count
        var = total_sq_sum / total_count - mean * mean
        var = torch.clamp(var, min=0.0)

        weight = self.weight
        bias = self.bias

        def group_norm_func(x, mean=mean, var=var, weight=weight, bias=bias):
            return custom_group_norm(x, num_groups, mean, var, weight, bias, 1e-6)

        return group_norm_func


class TiledVAE:
    def __init__(self, net, tile_size, is_decoder, to_gpu=False):
        self.net = net
        self.tile_size = tile_size
        self.is_decoder = is_decoder

        self.to_gpu = to_gpu
        self.pad = 11 if is_decoder else 32

    def __call__(self, x):
        B, C, H, W = x.shape

        param = next(self.net.parameters())
        original_device = param.device
        original_dtype = param.dtype

        target_device = devices.get_optimal_device() if self.to_gpu else original_device

        try:
            self.net.to(device=target_device, dtype=torch.float32)
            x = x.to(device=target_device, dtype=torch.float32)

            if max(H, W) <= self.pad * 2 + self.tile_size:
                print("[Tiled VAE]: the input size is tiny and unnecessary to tile.")
                out = self.net.original_forward(x)
            else:
                print("[Tiled VAE]: the input size is necessary to tile.")
                out = self.vae_tile_forward(x)

            if out.dtype != original_dtype:
                out = out.to(dtype=original_dtype)
            return out

        finally:
            self.net.to(device=original_device, dtype=original_dtype)

    def get_best_tile_size(self, lowerbound, upperbound):
        divider = 32
        while divider >= 2:
            remainer = lowerbound % divider
            if remainer == 0:
                return lowerbound
            candidate = lowerbound - remainer + divider
            if candidate <= upperbound:
                return candidate
            divider //= 2
        return lowerbound

    def split_tiles(self, h, w):
        tile_input_bboxes, tile_output_bboxes, tile_core_bboxes = [], [], []
        tile_size = self.tile_size
        pad = self.pad

        num_height_tiles = math.ceil((h - 2 * pad) / tile_size)
        num_width_tiles = math.ceil((w - 2 * pad) / tile_size)
        num_height_tiles = max(num_height_tiles, 1)
        num_width_tiles = max(num_width_tiles, 1)

        real_tile_height = math.ceil((h - 2 * pad) / num_height_tiles)
        real_tile_width = math.ceil((w - 2 * pad) / num_width_tiles)
        real_tile_height = self.get_best_tile_size(real_tile_height, tile_size)
        real_tile_width = self.get_best_tile_size(real_tile_width, tile_size)

        print(
            f"[Tiled VAE]: split to {num_height_tiles}x{num_width_tiles} = {num_height_tiles * num_width_tiles} tiles. "
            f"Optimal tile size {real_tile_width}x{real_tile_height}, original tile size {tile_size}x{tile_size}"
        )

        for i in range(num_height_tiles):
            for j in range(num_width_tiles):
                input_bbox = [
                    pad + j * real_tile_width,
                    min(pad + (j + 1) * real_tile_width, w),
                    pad + i * real_tile_height,
                    min(pad + (i + 1) * real_tile_height, h),
                ]

                core_bbox = [
                    input_bbox[0] if input_bbox[0] > pad else 0,
                    input_bbox[1] if input_bbox[1] < w - pad else w,
                    input_bbox[2] if input_bbox[2] > pad else 0,
                    input_bbox[3] if input_bbox[3] < h - pad else h,
                ]

                output_bbox = [x * 8 if self.is_decoder else x // 8 for x in core_bbox]

                tile_core_bboxes.append(core_bbox)
                tile_output_bboxes.append(output_bbox)

                tile_input_bboxes.append(
                    [
                        max(0, input_bbox[0] - pad),
                        min(w, input_bbox[1] + pad),
                        max(0, input_bbox[2] - pad),
                        min(h, input_bbox[3] + pad),
                    ]
                )

        return tile_input_bboxes, tile_output_bboxes, tile_core_bboxes

    def _attn_scale_num_den(self):
        if self.is_decoder:
            return 1, 1
        else:
            return 1, 8

    @torch.no_grad()
    def _run_global_attention(
        self,
        tiles,
        task_queues,
        in_bboxes,
        core_bboxes,
        full_h,
        full_w,
        attn_net,
        device,
    ):
        scale_num, scale_den = self._attn_scale_num_den()

        merged = None
        active_indices = []

        for i in range(len(tiles)):
            if tiles[i] is None:
                continue
            if len(task_queues[i]) == 0:
                continue

            tile = tiles[i].to(device=device, dtype=torch.float32)
            core_bbox_scaled = scale_bbox(core_bboxes[i], scale_num, scale_den)
            valid = crop_valid_region_with_scale(
                tile, in_bboxes[i], core_bbox_scaled, scale_num, scale_den
            )

            if merged is None:
                merged = torch.zeros(
                    (tile.shape[0], tile.shape[1], full_h, full_w),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=False,
                )

            merged[
                :,
                :,
                core_bbox_scaled[2] : core_bbox_scaled[3],
                core_bbox_scaled[0] : core_bbox_scaled[1],
            ] = valid
            active_indices.append(i)

        if merged is None:
            return 0

        attn_out = run_attention_layer(attn_net, merged)

        for i in active_indices:
            input_bbox_scaled = scale_bbox(in_bboxes[i], scale_num, scale_den)
            tiles[i] = attn_out[
                :,
                :,
                input_bbox_scaled[2] : input_bbox_scaled[3],
                input_bbox_scaled[0] : input_bbox_scaled[1],
            ].cpu()

        del merged, attn_out
        devices.torch_gc()
        gc.collect()
        return len(active_indices)

    @torch.no_grad()
    def vae_tile_forward(self, z):
        device = next(self.net.parameters()).device
        net = self.net
        tile_size = self.tile_size
        is_decoder = self.is_decoder

        z = z.detach().float()

        N, height, width = z.shape[0], z.shape[2], z.shape[3]
        net.last_z_shape = z.shape

        print(
            f"[Tiled VAE]: input_size: {z.shape}, tile_size: {tile_size}, padding: {self.pad}"
        )

        in_bboxes, out_bboxes, core_bboxes = self.split_tiles(height, width)

        tiles = []
        for input_bbox in in_bboxes:
            tile = z[
                :, :, input_bbox[2] : input_bbox[3], input_bbox[0] : input_bbox[1]
            ].cpu()
            tiles.append(tile)

        num_tiles = len(tiles)
        num_completed = 0

        single_task_queue = build_task_queue(net, is_decoder)
        task_queues = [clone_task_queue(single_task_queue) for _ in range(num_tiles)]

        result = None
        del z

        pbar = tqdm(
            total=num_tiles * len(task_queues[0]),
            desc=f"[Tiled VAE]: Executing {'Decoder' if is_decoder else 'Encoder'} Task Queue: ",
        )

        forward = True

        while True:
            group_norm_param = GroupNormParam()
            sync_mode = None
            pending_attn_net = None

            for i in range(num_tiles) if forward else reversed(range(num_tiles)):
                task_queue = task_queues[i]

                if tiles[i] is None:
                    continue
                if len(task_queue) == 0:
                    continue

                tile = tiles[i].to(device=device, dtype=torch.float32)

                while len(task_queue) > 0:
                    task = task_queue.pop(0)

                    if task[0] == "pre_norm":
                        if sync_mode is None:
                            sync_mode = "group_norm"
                        elif sync_mode != "group_norm":
                            raise RuntimeError(
                                f"Mixed sync mode detected: {sync_mode} vs group_norm"
                            )

                        group_norm_param.add_tile(tile, task[1])
                        break

                    elif task[0] == "attn_sync":
                        if sync_mode is None:
                            sync_mode = "attention"
                        elif sync_mode != "attention":
                            raise RuntimeError(
                                f"Mixed sync mode detected: {sync_mode} vs attention"
                            )

                        if pending_attn_net is None:
                            pending_attn_net = task[1]
                        break

                    elif task[0] == "store_res":
                        task_id = 0
                        res = task[1](tile)
                        res = res.cpu()
                        while task_queue[task_id][0] != "add_res":
                            task_id += 1
                        task_queue[task_id][1] = res

                    elif task[0] == "add_res":
                        tile += task[1].to(device=device, dtype=torch.float32)
                        task[1] = None

                    else:
                        tile = task[1](tile)

                    pbar.update(1)

                if len(task_queue) == 0:
                    tiles[i] = None
                    num_completed += 1

                    if result is None:
                        result = torch.zeros(
                            (
                                N,
                                tile.shape[1],
                                height * 8 if is_decoder else height // 8,
                                width * 8 if is_decoder else width // 8,
                            ),
                            device=device,
                            dtype=torch.float32,
                            requires_grad=False,
                        )

                    result[
                        :,
                        :,
                        out_bboxes[i][2] : out_bboxes[i][3],
                        out_bboxes[i][0] : out_bboxes[i][1],
                    ] = crop_valid_region(tile, in_bboxes[i], out_bboxes[i], is_decoder)
                    del tile

                elif sync_mode == "attention":
                    tiles[i] = tile.cpu()
                    del tile

                elif i == num_tiles - 1 and forward:
                    forward = False
                    tiles[i] = tile

                elif i == 0 and not forward:
                    forward = True
                    tiles[i] = tile

                else:
                    tiles[i] = tile.cpu()
                    del tile

            if num_completed == num_tiles:
                break

            if sync_mode == "group_norm":
                group_norm_func = group_norm_param.summary()
                if group_norm_func is not None:
                    for i in range(num_tiles):
                        if len(task_queues[i]) > 0:
                            task_queues[i].insert(0, ("apply_norm", group_norm_func))

            elif sync_mode == "attention":
                if is_decoder:
                    full_h, full_w = height, width
                else:
                    full_h, full_w = height // 8, width // 8

                n_attn_tiles = self._run_global_attention(
                    tiles=tiles,
                    task_queues=task_queues,
                    in_bboxes=in_bboxes,
                    core_bboxes=core_bboxes,
                    full_h=full_h,
                    full_w=full_w,
                    attn_net=pending_attn_net,
                    device=device,
                )
                pbar.update(n_attn_tiles)

            else:
                raise RuntimeError(
                    "No synchronization point found, but tiled VAE has not finished."
                )

        pbar.close()
        return result
