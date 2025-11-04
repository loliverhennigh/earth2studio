#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetch a Planetary Computer MODIS Fire tile and save quicklook PNGs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from earth2studio.data.planetary_computer import PlanetaryComputerMODISFire


def _plot_layer(array: np.ndarray, title: str, cmap: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.imshow(array, cmap=cmap, origin="upper")
    plt.colorbar(mesh, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path.name}")


def main() -> None:
    """Download a MODIS Fire tile and render FireMask/FRP/QA quicklooks."""
    datasource = PlanetaryComputerMODISFire(cache=True, verbose=True)
    request_time = datetime(2024, 6, 1, tzinfo=timezone.utc)
    data = datasource(time=request_time, variable=["fire_mask", "max_frp", "qa"])
    tile = data.isel(time=0)

    output_dir = Path(__file__).parent
    _plot_layer(
        tile.sel(variable="fire_mask").values,
        f"MODIS FireMask {request_time.date().isoformat()}",
        cmap="tab20",
        output_path=output_dir / "mpc_modis_fire_mask.png",
    )
    _plot_layer(
        tile.sel(variable="max_frp").values,
        f"MODIS Max FRP {request_time.date().isoformat()}",
        cmap="inferno",
        output_path=output_dir / "mpc_modis_fire_frp.png",
    )
    _plot_layer(
        tile.sel(variable="qa").values,
        f"MODIS QA {request_time.date().isoformat()}",
        cmap="plasma",
        output_path=output_dir / "mpc_modis_fire_qa.png",
    )


if __name__ == "__main__":
    main()

