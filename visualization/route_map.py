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

try:
    from scipy.spatial import ConvexHull
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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

    # ── Zone Partition Map (N/S/E/W) ─────────────────────────────────────────

    ZONE_COLORS = {
        "North" : "#4363d8",   # blue
        "South" : "#e6194b",   # red
        "East"  : "#3cb44b",   # green
        "West"  : "#f58231",   # orange
        "CBD"   : "#911eb4",   # purple
    }

    def save_zone_partition_map(
        self,
        filename : str = "zone_partition_map.html",
        title    : str = "City Zone Partition — N / S / E / W",
    ) -> str:
        """
        Draw the city divided into 4 geographic quadrants (N/S/E/W) using the
        city centre as the dividing point. Stops are coloured by their assigned
        zone. No route connections are drawn.
        """
        if not _FOLIUM_AVAILABLE:
            logger.warning("folium unavailable — skipping zone partition map")
            return ""

        stops  = self.graph.stops
        clat   = stops["stop_lat"].mean()
        clon   = stops["stop_lon"].mean()
        center = [clat, clon]

        min_lat = stops["stop_lat"].min() - 0.02
        max_lat = stops["stop_lat"].max() + 0.02
        min_lon = stops["stop_lon"].min() - 0.02
        max_lon = stops["stop_lon"].max() + 0.02

        fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

        # Title
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:14px;font-weight:bold;">
          {title}
        </div>"""
        fmap.get_root().html.add_child(folium.Element(title_html))

        # ── Draw 4 quadrant rectangles ────────────────────────────────────────
        quadrants = {
            "North": [[clat, min_lon], [max_lat, max_lon]],
            "South": [[min_lat, min_lon], [clat, max_lon]],
            "East" : [[min_lat, clat], [max_lat, max_lon]],   # overdrawn below
            "West" : [[min_lat, min_lon], [max_lat, clat]],   # overdrawn below
        }
        # Use proper lat/lon rectangles (SW corner → NE corner)
        quadrant_bounds = {
            "North": {"bounds": [[clat, min_lon], [max_lat, max_lon]], "label_lat": (clat + max_lat) / 2, "label_lon": (min_lon + max_lon) / 2},
            "South": {"bounds": [[min_lat, min_lon], [clat,   max_lon]], "label_lat": (min_lat + clat) / 2, "label_lon": (min_lon + max_lon) / 2},
            "East" : {"bounds": [[min_lat, clon],   [max_lat, max_lon]], "label_lat": (min_lat + max_lat) / 2, "label_lon": (clon + max_lon) / 2},
            "West" : {"bounds": [[min_lat, min_lon],[max_lat, clon]],    "label_lat": (min_lat + max_lat) / 2, "label_lon": (min_lon + clon) / 2},
        }

        for zone, info in quadrant_bounds.items():
            color = self.ZONE_COLORS.get(zone, "#aaaaaa")
            folium.Rectangle(
                bounds       = info["bounds"],
                color        = color,
                weight       = 2,
                opacity      = 0.5,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.10,
                tooltip      = zone,
            ).add_to(fmap)

            # Zone label marker
            folium.Marker(
                location=[info["label_lat"], info["label_lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:18px;font-weight:bold;'
                         f'color:{color};text-shadow:1px 1px 2px white;">{zone}</div>',
                    icon_size=(80, 30),
                    icon_anchor=(40, 15),
                ),
            ).add_to(fmap)

        # ── Division cross-hair lines ─────────────────────────────────────────
        # Horizontal divider (East-West axis)
        folium.PolyLine([[clat, min_lon], [clat, max_lon]],
                        color="#333333", weight=1.5, opacity=0.5,
                        dash_array="6 4").add_to(fmap)
        # Vertical divider (North-South axis)
        folium.PolyLine([[min_lat, clon], [max_lat, clon]],
                        color="#333333", weight=1.5, opacity=0.5,
                        dash_array="6 4").add_to(fmap)

        # ── Plot stops coloured by zone, no connections ───────────────────────
        for _, row in stops.iterrows():
            zone  = row.get("zone", "unknown")
            color = self.ZONE_COLORS.get(zone, "#888888")
            folium.CircleMarker(
                location     = [row["stop_lat"], row["stop_lon"]],
                radius       = 6,
                color        = color,
                fill         = True,
                fill_opacity = 0.9,
                tooltip      = (f"{row.get('stop_name', row['stop_id'])} "
                                f"| zone: {zone} | demand: {row.get('demand', 0)}"),
            ).add_to(fmap)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_items = "".join(
            f'<div><span style="background:{c};width:16px;height:16px;'
            f'display:inline-block;border-radius:50%;margin-right:6px;"></span>{z}</div>'
            for z, c in self.ZONE_COLORS.items()
        )
        legend_html = f"""
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                    background:white;padding:10px;border-radius:8px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:13px;">
          <b>Zone</b><br>{legend_items}
        </div>"""
        fmap.get_root().html.add_child(folium.Element(legend_html))

        path = os.path.join(self.output_dir, filename)
        fmap.save(path)
        logger.info("Zone partition map saved: %s", path)
        return path

    # ── Zone Partition + GWO Routes Overlay ──────────────────────────────────

    def save_zone_optimized_map(
        self,
        routes   : List[Dict],
        algo_name: str = "CUDA-GWO",
        filename : str = "zone_optimized_map.html",
    ) -> str:
        """
        Overlay GWO-optimized routes on the N/S/E/W zone partition map.

        - Zone quadrants drawn as low-opacity coloured rectangles
        - Stops coloured by their zone (no connections)
        - Optimized routes drawn on top as coloured polylines
        - Route start (★) and end (■) stops labelled
        """
        if not _FOLIUM_AVAILABLE:
            logger.warning("folium unavailable — skipping zone optimized map")
            return ""

        stops  = self.graph.stops
        clat   = stops["stop_lat"].mean()
        clon   = stops["stop_lon"].mean()
        center = [clat, clon]

        min_lat = stops["stop_lat"].min() - 0.02
        max_lat = stops["stop_lat"].max() + 0.02
        min_lon = stops["stop_lon"].min() - 0.02
        max_lon = stops["stop_lon"].max() + 0.02

        fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

        # Title
        title_html = f"""
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:8px 16px;
                    border-radius:8px;box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:14px;font-weight:bold;">
          {algo_name} — Optimized Routes on Zone Partition
        </div>"""
        fmap.get_root().html.add_child(folium.Element(title_html))

        # ── Zone quadrant rectangles ──────────────────────────────────────────
        quadrant_bounds = {
            "North": {"bounds": [[clat, min_lon], [max_lat, max_lon]],
                      "label_lat": (clat + max_lat) / 2, "label_lon": (min_lon + max_lon) / 2},
            "South": {"bounds": [[min_lat, min_lon], [clat, max_lon]],
                      "label_lat": (min_lat + clat) / 2, "label_lon": (min_lon + max_lon) / 2},
            "East" : {"bounds": [[min_lat, clon], [max_lat, max_lon]],
                      "label_lat": (min_lat + max_lat) / 2, "label_lon": (clon + max_lon) / 2},
            "West" : {"bounds": [[min_lat, min_lon], [max_lat, clon]],
                      "label_lat": (min_lat + max_lat) / 2, "label_lon": (min_lon + clon) / 2},
        }
        for zone, info in quadrant_bounds.items():
            color = self.ZONE_COLORS.get(zone, "#aaaaaa")
            folium.Rectangle(
                bounds       = info["bounds"],
                color        = color,
                weight       = 2,
                opacity      = 0.4,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.07,
                tooltip      = zone,
            ).add_to(fmap)
            folium.Marker(
                location=[info["label_lat"], info["label_lon"]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:16px;font-weight:bold;'
                         f'color:{color};text-shadow:1px 1px 2px white;">{zone}</div>',
                    icon_size=(80, 30), icon_anchor=(40, 15),
                ),
            ).add_to(fmap)

        # Divider lines
        folium.PolyLine([[clat, min_lon], [clat, max_lon]],
                        color="#555", weight=1.2, opacity=0.4,
                        dash_array="6 4").add_to(fmap)
        folium.PolyLine([[min_lat, clon], [max_lat, clon]],
                        color="#555", weight=1.2, opacity=0.4,
                        dash_array="6 4").add_to(fmap)

        # ── Collect which routes serve each stop (for tooltips) ──────────────
        stop_routes: Dict[int, List[str]] = {}
        for r_idx, route in enumerate(routes):
            name = route.get("route_id", f"R{r_idx}")
            for s in route["stop_sequence"]:
                stop_routes.setdefault(int(s), []).append(name)

        # ── For each zone: connect ALL stops via nearest-neighbour path ────────
        zones_present = stops["zone"].unique()
        for zone in zones_present:
            zone_stops = stops[stops["zone"] == zone]
            if len(zone_stops) < 2:
                continue

            color      = self.ZONE_COLORS.get(str(zone), "#888888")
            zone_idx   = zone_stops.index.tolist()
            lats       = zone_stops["stop_lat"].values
            lons       = zone_stops["stop_lon"].values

            # Greedy nearest-neighbour order through all stops in the zone
            visited  = [False] * len(zone_idx)
            order    = [0]
            visited[0] = True
            for _ in range(len(zone_idx) - 1):
                cur   = order[-1]
                best_d, best_j = np.inf, -1
                for j in range(len(zone_idx)):
                    if visited[j]:
                        continue
                    d = (lats[cur] - lats[j])**2 + (lons[cur] - lons[j])**2
                    if d < best_d:
                        best_d, best_j = d, j
                order.append(best_j)
                visited[best_j] = True

            coords = [[lats[i], lons[i]] for i in order]
            route_names = set(
                r for idx in zone_idx for r in stop_routes.get(idx, [])
            )
            folium.PolyLine(
                locations = coords,
                color     = color,
                weight    = 4,
                opacity   = 0.85,
                tooltip   = f"{zone} zone | routes: {', '.join(sorted(route_names)) or 'none'}",
            ).add_to(fmap)

        # ── Stops — same style as route_map, coloured by zone ─────────────────
        for idx, row in stops.iterrows():
            zone   = str(row.get("zone", "unknown"))
            color  = self.ZONE_COLORS.get(zone, "#888888")
            routes_here = stop_routes.get(int(idx), [])
            marker = "●"
            folium.CircleMarker(
                location     = [row["stop_lat"], row["stop_lon"]],
                radius       = 5,
                color        = color,
                fill         = True,
                fill_opacity = 0.90,
                tooltip      = (
                    f"{marker} {row.get('stop_name', row['stop_id'])} "
                    f"| zone: {zone} "
                    f"| demand: {row.get('demand', 0)} "
                    f"| routes: {', '.join(routes_here) if routes_here else 'none'}"
                ),
            ).add_to(fmap)

        # ── Legend ────────────────────────────────────────────────────────────
        zone_items = "".join(
            f'<div><span style="background:{c};width:14px;height:14px;'
            f'display:inline-block;border-radius:50%;margin-right:5px;"></span>{z}</div>'
            for z, c in self.ZONE_COLORS.items()
        )
        legend_html = f"""
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
                    background:white;padding:10px;border-radius:8px;
                    box-shadow:0 2px 6px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:12px;max-height:340px;overflow-y:auto;">
          <b>Zones</b><br>{zone_items}
          <hr style="margin:5px 0">
          <div style="font-size:11px;color:#555;">
            Lines connect stops<br>within the same zone only
          </div>
        </div>"""
        fmap.get_root().html.add_child(folium.Element(legend_html))

        path = os.path.join(self.output_dir, filename)
        fmap.save(path)
        logger.info("Zone-optimized map saved: %s", path)
        return path

    # ── Folium Interactive Map ────────────────────────────────────────────────

    def save_interactive_map(
        self,
        routes         : List[Dict],
        filename       : str                    = "route_map.html",
        title          : str                    = "Optimized Transit Routes",
        cluster_labels : Optional[np.ndarray]   = None,
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

        # ── K-means cluster boundaries (low-opacity convex hulls) ────────────
        if cluster_labels is not None:
            self._draw_cluster_boundaries(fmap, stops, cluster_labels)

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

    def _draw_cluster_boundaries(
        self,
        fmap          : "folium.Map",
        stops         : pd.DataFrame,
        cluster_labels: np.ndarray,
    ):
        """
        Draw a low-opacity convex-hull polygon around each K-means cluster.
        Falls back to a bounding-circle if scipy is unavailable or cluster < 3 stops.
        """
        k = int(cluster_labels.max()) + 1
        for c in range(k):
            idx    = np.where(cluster_labels == c)[0]
            color  = self.ROUTE_COLORS[c % len(self.ROUTE_COLORS)]
            lats   = stops.iloc[idx]["stop_lat"].values
            lons   = stops.iloc[idx]["stop_lon"].values

            if len(idx) >= 3 and _SCIPY_AVAILABLE:
                pts  = np.column_stack([lats, lons])
                try:
                    hull      = ConvexHull(pts)
                    hull_pts  = [[lats[i], lons[i]] for i in hull.vertices]
                    hull_pts.append(hull_pts[0])   # close the polygon
                    folium.Polygon(
                        locations   = hull_pts,
                        color       = color,
                        weight      = 1.5,
                        opacity     = 0.35,
                        fill        = True,
                        fill_color  = color,
                        fill_opacity= 0.07,
                        tooltip     = f"Cluster {c}",
                    ).add_to(fmap)
                except Exception:
                    pass   # degenerate hull — skip silently
            else:
                # Fallback: circle centred on cluster centroid
                clat, clon = lats.mean(), lons.mean()
                # Radius = max distance from centroid (approx in degrees → metres)
                dists = np.sqrt((lats - clat)**2 + (lons - clon)**2)
                radius_m = float(dists.max()) * 111_000   # 1° ≈ 111 km
                folium.Circle(
                    location    = [clat, clon],
                    radius      = max(radius_m, 300),
                    color       = color,
                    weight      = 1.5,
                    opacity     = 0.35,
                    fill        = True,
                    fill_color  = color,
                    fill_opacity= 0.07,
                    tooltip     = f"Cluster {c}",
                ).add_to(fmap)

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
