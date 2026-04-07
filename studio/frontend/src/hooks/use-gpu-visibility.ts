// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState, useEffect } from "react";

interface GpuDevice {
  index: number;
  name: string;
  memoryTotalGb: number;
}

export interface GpuVisibility {
  available: boolean;
  devices: GpuDevice[];
  parentVisibleGpuIds: number[];
  isMultiGpu: boolean;
}

const EMPTY: GpuVisibility = {
  available: false,
  devices: [],
  parentVisibleGpuIds: [],
  isMultiGpu: false,
};

let cached: GpuVisibility | null = null;
let fetchPromise: Promise<GpuVisibility> | null = null;

async function fetchOnce(): Promise<GpuVisibility> {
  if (cached) return cached;
  if (fetchPromise) return fetchPromise;

  fetchPromise = fetch("/api/system/gpu-visibility")
    .then(async (res) => {
      if (!res.ok) return EMPTY;
      const data = await res.json();
      const devices: GpuDevice[] = (data.devices ?? []).map(
        (d: { index: number; name: string; memory_total_gb: number }) => ({
          index: d.index,
          name: d.name,
          memoryTotalGb: d.memory_total_gb,
        }),
      );
      const result: GpuVisibility = {
        available: data.available ?? false,
        devices,
        parentVisibleGpuIds: data.parent_visible_gpu_ids ?? [],
        isMultiGpu: devices.length > 1,
      };
      cached = result;
      return result;
    })
    .catch(() => {
      fetchPromise = null;
      return EMPTY;
    });

  return fetchPromise;
}

export function useGpuVisibility(): GpuVisibility {
  const [info, setInfo] = useState<GpuVisibility>(cached ?? EMPTY);

  useEffect(() => {
    if (cached) return;
    let cancelled = false;
    fetchOnce().then((result) => {
      if (!cancelled) setInfo(result);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return info;
}
