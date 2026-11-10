# Evidence for FlipTrace-GC

Run `python -S -B verify_evidence.py` from the release root. The verifier uses the Python standard library and the released cache policies. It recomputes the paired effects and acceptance gates, calculates held-out mIoU from the confusion counts, and replays the four recorded access streams. The archived measurements cover GCNet-S on CamVid on one NVIDIA GeForce RTX 4060 Laptop GPU. The verifier checks the numerical evidence and policy replay without running training, collecting new timing measurements, or performing inference.

## Paired timing

[`paired-timing.json`](paired-timing.json) contains eight 400-update pairs from the manuscript, with each arm's recorded training seconds, validation mIoU, peak GPU allocation and cache payload. It also records the execution order, paired wall-clock budget and elapsed time, and the prespecified gates. The underlying run configurations define the protocol and gates. The comparison records supply the paired measurements and gate outcomes, which were checked against the 16 individual result records. Validation mIoU was also recomputed from each individual result's confusion matrix during export.

| Pair | Recorded comparison | Trace |
| --- | --- | --- |
| R1 | Expanded recomputation then schedule | A/3407 |
| R2 | Expanded schedule then recomputation | A/304 |
| L1 | Expanded LRU then schedule | A/304 |
| L2 | Expanded schedule then LRU | B/3407 |
| F | Folded recomputation then schedule | A/3407 |
| E1 | Expanded recomputation then native-precision eager prebuilding | A/3407 |
| E2 | Expanded eager prebuilding then recomputation | A/304 |
| E3 | Expanded recomputation then eager prebuilding on the official training split | Official/3407 |

`train_seconds` spans the timed segment after warm-up and reset through the end of training. Cache setup is already included in this interval. Validation and model saving are outside it. `cache_host_bytes` is the peak cache tensor payload, and `worker_peak_host_working_set_bytes`, when recorded, covers the worker process. `train_peak_allocated_bytes` is peak training allocation as reported by PyTorch. The F record also includes the shared folding and verification time, measured separately from either arm.

The timing gate uses `1 - candidate_seconds / reference_seconds`. The mIoU change is candidate minus reference in percentage points. The GPU gate uses candidate peak allocation divided by reference peak allocation. The 64 MiB cache payload cap applies to both arms of the on-demand pairs. The folded comparison and the official-split eager comparison missed their respective time gates.

## Held-out confusion counts

[`heldout-confusions.json`](heldout-confusions.json) includes the four aggregate 11 × 11 confusion matrices from the endpoints of L1 and L2. The original endpoint identifiers were checked against the locked evaluation plan. Rows count ground-truth classes and columns count predictions. Void pixels are excluded. All four records cover the same 233 official test images and 38,719,017 valid pixels. The verifier checks the shared ground-truth marginals and computes global mIoU as the unweighted mean of per-class intersection over union across nonzero unions. It then applies the 0.10 percentage-point absolute tolerance to the schedule-minus-LRU contrast within each pair.

## Exact access replay

[`schedule-traces.json`](schedule-traces.json) contains four subset-and-seed streams. Each `batches` array holds 400 batches in completed-update order. Each view key consists of the zero-based index in the ordered training list and a Boolean horizontal-flip flag. The order of views within a batch is preserved because LRU uses that order. A and B use their frozen subset list orders. The official training trace uses the 367 RGB basenames extracted from the original SegNet CamVid training list and sorted lexicographically with case preserved.

The export reconstructed the original index stream by shuffling indices separately for each epoch with a Python random generator seeded with the run seed plus the zero-based epoch. A second generator initialized with the run seed supplied the per-draw flip flags using a 0.5 threshold. Every reconstructed batch was checked against the ordered filename and flip records in all 16 archived arms. For the official stream, the recovered index-to-basename mapping also matched the frozen digest of the ordered basenames. The same seed yields identical numeric index-and-flip streams on A and B because the subsets are the same size. Their indices refer to different frames.

The `expected_replay` fields give full-batch hits and net view writes from the original policy functions at 45 slots, starting empty. A full-batch hit requires every requested key to be resident before the step. Writes count newly admitted keys retained at the end of the step. The verifier recomputes both counters for each policy and compares them with the observed counters wherever that policy ran. LRU on A/3407 and both bounded policies on the official training stream are replay-only. E3's measured eager experiment prebuilt 734 views, of which the archived stream accessed 575.
