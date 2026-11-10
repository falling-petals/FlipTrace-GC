# FlipTrace-GC

FlipTrace-GC replays frozen GCNet-S features during a fixed sequence of CamVid adaptation updates. It uses the next access to each image-and-flip key to decide which views to retain in a 45-slot host cache. The study measures elapsed training time at the same update count against immediate recomputation and same-capacity LRU.

This release contains the project's cache and replay code, the shared LoRA and loss implementation, the ordered training subsets, and derived records from the paper. The GCNet-S implementation, CamVid images and labels, and Cityscapes initialization come from the original providers. Historical training and evaluation runners and their checkpoints remain in the project archive.

## Check the archived evidence

From this directory, run with Python 3.12.

```sh
python verify_evidence.py
```

The checker uses the standard library and the packaged cache policies. It replays the logged draw streams, checks hit and write counts against the recorded bounded runs, calculates the relative changes from the paired training times, and reconstructs held-out mIoU from the confusion matrices. The training times are archived measurements from one RTX 4060 Laptop GPU. The checker does not remeasure elapsed time or evaluate a model. [Evidence format and provenance](evidence/README.md)

## Integrate into a training loop

Use the replay mixin with an external GCNet-S port that implements the three split hooks documented in the integration guide. That guide also records the dataset, pretrained checkpoint, label convention, evaluation protocol, and commands for the archived workspace. [Integration and matched-run protocol](INTEGRATION.md)

- [`fliptrace_gc/policy.py`](fliptrace_gc/policy.py) contains the schedule-aware and LRU policies from the experiments, with no PyTorch dependency.
- [`fliptrace_gc/replay.py`](fliptrace_gc/replay.py) contains the bounded cache, the native-precision packing and replay methods, and the eager comparison path. The original per-image power check in the eager path is exposed as an optional callback.
- [`fliptrace_gc/adaptation.py`](fliptrace_gc/adaptation.py) contains the shared rank-12 convolutional adaptation and two-head CE/Dice used for both arms of each pair.
- [Subset A](subset-a.txt) and [subset B](subset-b.txt) list the selected CamVid frame basenames in training-index order.

Use the same fixed normalized input and the same horizontal flip each time a key recurs. The replay path assumes the entire cached prefix and its normalization statistics stay fixed. It preserves the original full-batch shape on misses and checks the first full replay against the online features and both logits. Keep one cache per run and create it after restoring the warm-up state.

The study's default capacity is 64 MiB of cached CPU tensor payload. For the archived input shape, 45 views occupy 63.28125 MiB. The paired LRU comparison used the same capacity. This limit tracks the stored tensors, while total process memory is recorded separately.

The code repository is [falling-petals/FlipTrace-GC](https://github.com/falling-petals/FlipTrace-GC).

## License

The FlipTrace-GC software and its associated documentation are licensed under the [MIT License](LICENSE). The dataset, pretrained weights, and external software are subject to their respective providers' terms. The included subset lists and archived evidence are described in the [evidence provenance notes](evidence/README.md).
