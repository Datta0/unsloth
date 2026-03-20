// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useState } from "react";

export interface GpuDevice {
  index: number;
  name: string;
  memory_total_gb: number;
}

export interface GpuVisibility {
  available: boolean;
  backend_cuda_visible_devices: string | null;
  parent_visible_gpu_ids: number[];
  devices: GpuDevice[];
}

const DEFAULT: GpuVisibility = {
  available: false,
  backend_cuda_visible_devices: null,
  parent_visible_gpu_ids: [],
  devices: [],
};

/**
 * Fetch GPU visibility info from the backend once on mount.
 * Returns the list of available GPUs with their names and VRAM.
 */
export function useGpuVisibility(): GpuVisibility {
  const [data, setData] = useState<GpuVisibility>(DEFAULT);

  useEffect(() => {
    let cancelled = false;

    async function fetch() {
      try {
        const res = await authFetch("/api/system/gpu-visibility");
        if (!res.ok || cancelled) return;
        const json = (await res.json()) as GpuVisibility;
        if (!cancelled) setData(json);
      } catch {
        // Silently ignore — single GPU fallback
      }
    }

    void fetch();
    return () => {
      cancelled = true;
    };
  }, []);

  return data;
}
