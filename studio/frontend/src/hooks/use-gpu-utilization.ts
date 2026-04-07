// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { authFetch } from "@/features/auth";

export interface GpuUtilization {
  gpu_utilization_pct: number | null;
  temperature_c: number | null;
  vram_used_gb: number | null;
  vram_total_gb: number | null;
  vram_utilization_pct: number | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  power_utilization_pct: number | null;
}

export interface GpuDeviceMetrics extends GpuUtilization {
  index: number;
}

const DEFAULT: GpuUtilization = {
  gpu_utilization_pct: null,
  temperature_c: null,
  vram_used_gb: null,
  vram_total_gb: null,
  vram_utilization_pct: null,
  power_draw_w: null,
  power_limit_w: null,
  power_utilization_pct: null,
};

export interface GpuUtilizationResult {
  /** Aggregate metrics (backwards-compatible with single-GPU consumers). */
  aggregate: GpuUtilization;
  /** Per-device metrics. Length > 1 means multi-GPU. */
  devices: GpuDeviceMetrics[];
}

const EMPTY_RESULT: GpuUtilizationResult = {
  aggregate: DEFAULT,
  devices: [],
};

export function useGpuUtilization(
  enabled: boolean,
  intervalMs = 10_000,
): GpuUtilizationResult {
  const [result, setResult] = useState<GpuUtilizationResult>(EMPTY_RESULT);

  useEffect(() => {
    if (!enabled) {
      setResult(EMPTY_RESULT);
      return;
    }

    let alive = true;

    async function poll() {
      try {
        const visRes = await authFetch("/api/train/hardware/visible");
        if (visRes.ok && alive) {
          const data = await visRes.json();
          const devices: GpuDeviceMetrics[] = data.devices ?? [];
          if (devices.length > 0) {
            setResult({ aggregate: devices[0], devices });
            return;
          }
        }
        const res = await authFetch("/api/train/hardware");
        if (res.ok && alive) {
          const agg = await res.json();
          setResult({
            aggregate: agg,
            devices: [{ index: 0, ...agg }],
          });
        }
      } catch {
        /* next poll will retry */
      }
    }

    void poll();
    const id = setInterval(() => void poll(), intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [enabled, intervalMs]);

  return result;
}
