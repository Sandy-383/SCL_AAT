"""
Interactive Route Map Visualizer.

Renders the optimized transit routes on an interactive Folium map.
Each route gets a distinct colour; stops are shown as circles with
tooltip information (stop name, demand, zone).

Output: HTML file that can be opened in any browser.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import folium
    from folium import plugins
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False
    logger.warning("folium not installed — interactive maps disabled")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class RouteMapVisualizer:
    """
    Renders transit route maps.

    Parameters
    ----------
    graph  : TransitGraph
    config : VIZ_CONFIG dict
    """

    ROUTE_COLORS = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
        "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    ]

    def __init__(self, graph, output_dir: str = "outputs"):
        self.graph      = graph
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Folium Interactive Map ────────────────────────────────────────────────

    def save_interactive_map(
        self,
        routes    : List[Dict],
        filename  : str  = "route_map.html",
        title     : str  = "Optimized Transit Routes",
    ) -> str:
        if not _FOLIUM_AVAILABLE:
            logger.warning("folium unavailable — skipping interactive map")
            return ""

        stops  = self.graph.stops
        center = [stops["stop_lat"].mean(), stops["stop_lon"].mean()]

        fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

        # Add title
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:14px;font-weight:bold;">
          {title}
        </div>"""
        fmap.get_root().html.add_child(folium.Element(title_html))

        # Plot all stops as small grey circles
        for _, row in stops.iterrows():
            folium.CircleMarker(
                location=[row["stop_lat"], row["stop_lon"]],
                radius=3,
                color="#888888",
                fill=True,
                fill_opacity=0.5,
                tooltip=f"{row.get('stop_name', row['stop_id'])} (demand: {row.get('demand',0)})",
            ).add_to(fmap)

        # Plot each route
        for r_idx, route in enumerate(routes):
            color = self.ROUTE_COLORS[r_idx % len(self.ROUTE_COLORS)]
            seq   = route["stop_sequence"]
            name  = route.get("route_id", f"R{r_idx}")
            hw    = route.get("headway_min", "?")
            nv    = route.get("num_vehicles", "?")

            # Draw polyline through stops
            coords = []
            for stop_idx in seq:
                if stop_idx < len(stops):
                    row = stops.iloc[stop_idx]
                    coords.append([row["stop_lat"], row["stop_lon"]])

            if len(coords) >= 2:
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=4,
                    opacity=0.85,
                    tooltip=f"{name} | headway: {hw} min | vehicles: {nv}",
                ).add_to(fmap)

            # Highlight stops on this route
            for i, stop_idx in enumerate(seq):
                if stop_idx >= len(stops):
                    continue
                row    = stops.iloc[stop_idx]
                marker = "★" if i == 0 else ("■" if i == len(seq)-1 else "●")
                folium.CircleMarker(
                    location=[row["stop_lat"], row["stop_lon"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.9,
                    tooltip=(f"{marker} {row.get('stop_name', stop_idx)} "
                             f"[{name}] stop {i+1}/{len(seq)}"),
                ).add_to(fmap)

        # Legend
        legend_html = self._build_legend_html(routes)
        fmap.get_root().html.add_child(folium.Element(legend_html))

        path = os.path.join(self.output_dir, filename)
        fmap.save(path)
        logger.info("Interactive map saved: %s", path)
        return path

    def _build_legend_html(self, routes: List[Dict]) -> str:
        items = ""
        for r_idx, route in enumerate(routes):
            color = self.ROUTE_COLORS[r_idx % len(self.ROUTE_COLORS)]
            name  = route.get("route_id", f"R{r_idx}")
            hw    = route.get("headway_min", "?")
            items += (f'<div><span style="background:{color};width:20px;height:10px;'
                      f'display:inline-block;margin-right:6px;"></span>'
                      f'{name} (hw: {hw} min)</div>')

        return f"""
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                    background:white;padding:10px;border-radius:8px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:12px;max-height:300px;overflow-y:auto;">
          <b>Routes</b><br>{items}
        </div>"""

    # ── Static Matplotlib Map ─────────────────────────────────────────────────

    def save_static_map(
        self,
        routes   : List[Dict],
        filename : str = "route_map_static.png",
        dpi      : int = 150,
    ) -> str:
        if not _MPL_AVAILABLE:
            return ""

        stops = self.graph.stops
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor("#f0f0f0")
        fig.patch.set_facecolor("white")

        # All stops
        ax.scatter(stops["stop_lon"], stops["stop_lat"],
                   s=8, c="#aaaaaa", zorder=2, alpha=0.6)

        patches = []
        for r_idx, route in enumerate(routes):
            color = self.ROUTE_COLORS[r_idx % len(self.ROUTE_COLORS)]
            seq   = route["stop_sequence"]
            name  = route.get("route_id", f"R{r_idx}")

            lats = [stops.iloc[s]["stop_lat"] for s in seq if s < len(stops)]
            lons = [stops.iloc[s]["stop_lon"] for s in seq if s < len(stops)]

            if len(lons) >= 2:
                ax.plot(lons, lats, "-o", color=color, linewidth=2,
                        markersize=4, zorder=3, alpha=0.85)

            patches.append(mpatches.Patch(color=color, label=name))

        ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=2)
        ax.set_title("Optimized Transit Routes", fontsize=14, fontweight="bold")
        ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info("Static map saved: %s", path)
        return path

    # ── Before/After Comparison ───────────────────────────────────────────────

    def save_comparison_map(
        self,
        routes_before : List[Dict],
        routes_after  : List[Dict],
        filename      : str = "comparison_map.png",
    ) -> str:
        if not _MPL_AVAILABLE:
            return ""

        stops = self.graph.stops
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for ax, routes, title in zip(
            axes,
            [routes_before, routes_after],
            ["Initial Routes", "Optimized Routes"]
        ):
            ax.set_facecolor("#f5f5f5")
            ax.scatter(stops["stop_lon"], stops["stop_lat"],
                       s=6, c="#bbbbbb", zorder=2, alpha=0.5)

            for r_idx, route in enumerate(routes):
                color = self.ROUTE_COLORS[r_idx % len(self.ROUTE_COLORS)]
                seq   = route["stop_sequence"]
                lats  = [stops.iloc[s]["stop_lat"] for s in seq if s < len(stops)]
                lons  = [stops.iloc[s]["stop_lon"] for s in seq if s < len(stops)]
                if len(lons) >= 2:
                    ax.plot(lons, lats, "-", color=color, linewidth=2.5,
                            alpha=0.85, zorder=3)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
            ax.grid(True, alpha=0.3)

        plt.suptitle("Route Optimization: Before vs After", fontsize=15, y=1.01)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Comparison map saved: %s", path)
        return path
