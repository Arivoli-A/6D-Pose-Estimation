# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE_DEFORMABLE_DETR in the LICENSES folder for details]
# ------------------------------------------------------------------------

import torch


def to_cuda(samples, targets, device):
    if isinstance(samples, list) and isinstance(samples[0], dict):
        samples_cuda = []
        for sample in samples:
            new_sample = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    new_sample[k] = v.to(device, non_blocking=True)
                elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                    new_sample[k] = [t.to(device, non_blocking=True) for t in v]
                else:
                    new_sample[k] = v  # leave non-tensor values unchanged
            samples_cuda.append(new_sample)
    else:
        samples_cuda = samples.to(device, non_blocking=True)
        
    if targets is not None:
        # Only write targets to cuda device if not None, otherwise return None
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return samples_cuda, targets


class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets = to_cuda(self.next_samples, self.next_targets, self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            if samples is not None:
                if isinstance(samples, list) and isinstance(samples[0], dict):
                    for s in samples:
                        for v in s.values():
                            if isinstance(v, torch.Tensor):
                                v.record_stream(torch.cuda.current_stream())
                            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                                for t in v:
                                    t.record_stream(torch.cuda.current_stream())
                else:
                    samples.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets = next(self.loader)
                samples, targets = to_cuda(samples, targets, self.device)
            except StopIteration:
                samples = None
                targets = None
        return samples, targets
