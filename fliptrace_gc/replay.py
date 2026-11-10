"""CUDA feature replay for a GCNet-S split supplied by the caller."""

from collections import OrderedDict
import math

import torch

from .policy import lru_prefix_reuse, schedule_prefix_reuse


SCHEDULE_CACHE_MAX_BYTES = 64 * 2**20
SCHEDULE_CACHE_SLOTS = 45


class PrefixReplayMixin:
    """Accept packed, frozen stage-6 inputs during cached updates."""

    def pack_prefix(self, features):
        if any(feature is None for feature in features):
            raise ValueError("Prefix cache requires both branch inputs and the auxiliary tap")
        self.replay_shapes = tuple(feature.shape[1:] for feature in features)
        self.replay_dtypes = tuple(feature.dtype for feature in features)
        self.replay_widths = tuple(math.prod(shape) * feature.element_size()
                                   for shape, feature in zip(self.replay_shapes, features, strict=True))
        return torch.cat([feature.contiguous().view(torch.uint8).flatten(1) for feature in features], dim=1)

    @torch.no_grad()
    def encode_prefix(self, images):
        self.replay_image_size = images.shape[-2:]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return self.pack_prefix(self.backbone.stage6_inputs(images))

    @torch.no_grad()
    def verify_replay(self, images, packed):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            online_features = self.backbone.stage6_inputs(images)
            for online, replayed in zip(online_features, self.unpack_prefix(packed), strict=True):
                torch.testing.assert_close(online, replayed, atol=0, rtol=0)
            for online, replayed in zip(self(images), self(packed), strict=True):
                torch.testing.assert_close(online, replayed, atol=0, rtol=0)

    @torch.no_grad()
    def build_prefix_cache(self, training, check_batch, *, before_encode=None):
        """Build the eager comparator cache, with an optional per-image check."""
        cache = {}
        self.replay_image_size = training[0][1].shape[-2:]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for index, (_, image, _) in enumerate(training):
                if before_encode is not None:
                    before_encode()
                features = self.backbone.stage6_inputs(torch.stack((image, image.flip(-1))).cuda())
                packed = self.pack_prefix(features).cpu()
                cache[index, False], cache[index, True] = packed.unbind()
            online_images = torch.stack([
                training[index][1].flip(-1) if flip else training[index][1] for index, flip in check_batch
            ]).cuda()
            replayed_inputs = torch.stack([cache[key] for key in check_batch]).cuda()
            self.verify_replay(online_images, replayed_inputs)
        torch.cuda.synchronize()
        return cache

    def unpack_prefix(self, packed):
        return tuple(part.contiguous().view(dtype).reshape(packed.shape[0], *shape)
                     for part, dtype, shape in zip(packed.split(self.replay_widths, dim=1),
                                                   self.replay_dtypes, self.replay_shapes, strict=True))

    def forward(self, x):
        if x.ndim != 2:
            return super().forward(x)
        features = self.backbone.stage6_output(*self.unpack_prefix(x))
        return self.decode_and_resize(features, self.replay_image_size)


class BoundedPrefixCache:
    def __init__(self, schedule, *, use_lru=False):
        self.recency = OrderedDict() if use_lru else None
        self.plan = None if use_lru else schedule_prefix_reuse(schedule, SCHEDULE_CACHE_SLOTS)
        self.entries = {}
        self.full_hits = 0
        self.writes = 0
        self.peak_bytes = 0

    def inputs(self, model, training, step, batch):
        if self.recency is None:
            full_hit, evict, admit = self.plan[step]
        else:
            full_hit, evict, admit = lru_prefix_reuse(self.recency, batch, SCHEDULE_CACHE_SLOTS)
        if full_hit:
            # Move individual entries to CUDA before stacking to avoid a second host payload.
            inputs = torch.stack([self.entries[key].cuda() for key in batch])
        if not full_hit or self.full_hits == 0:
            images = []
            for index, flip in batch:
                image = training[index][1]
                images.append(image.flip(-1) if flip else image)
            raw = torch.stack(images).cuda()
            if full_hit:
                model.verify_replay(raw, inputs)
            else:
                inputs = model.encode_prefix(raw) if admit else raw
        self.full_hits += int(full_hit)

        for key in evict:
            del self.entries[key]
        if admit:
            if inputs.shape[1] * SCHEDULE_CACHE_SLOTS > SCHEDULE_CACHE_MAX_BYTES:
                raise ValueError("Native views exceed the predeclared host cache budget")
            for key in admit:
                self.entries[key] = inputs[batch.index(key)].cpu()
            self.writes += len(admit)
            self.peak_bytes = max(self.peak_bytes, sum(view.nbytes for view in self.entries.values()))
        return inputs

    def finish(self):
        if self.recency is None:
            if self.entries:
                raise ValueError("Scheduled views were not released at their last use")
        elif self.recency.keys() != self.entries.keys():
            raise ValueError("LRU policy and cached views diverged")
        else:
            self.recency.clear()
        if not self.full_hits:
            raise ValueError("Bounded cache did not replay a batch")
        self.entries.clear()
