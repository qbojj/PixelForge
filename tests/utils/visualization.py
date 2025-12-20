"""Visualization utilities for rasterizer output"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Fragment:
    """Represents a rasterized fragment"""

    coord_pos: Tuple[int, int]
    color: Tuple[float, float, float, float]


class FragmentVisualizer:
    """Visualize rasterized fragments in various formats"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.canvas = np.zeros((height, width, 4), dtype=np.float32)  # RGBA
        self.depth = np.full((height, width), 1.0, dtype=np.float32)  # Depth buffer
        self.stencil = np.zeros((height, width), dtype=np.uint8)  # Stencil buffer

    def render(self, fragments: List[Fragment]):
        """Render fragments onto the canvas"""

        for frag in fragments:
            x, y = int(frag.coord_pos[0]), int(frag.coord_pos[1])

            # Bounds check
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue

            # TODO: Depth and stencil tests can be added here

            d_rgba = np.array(frag.color, dtype=np.float32)
            s_rgba = self.canvas[y, x]

            d_rgb = d_rgba[0:3]
            d_a = d_rgba[3]
            s_rgb = s_rgba[0:3]
            s_a = s_rgba[3]

            # Alpha blending
            def lerp(c1, c2, t):
                return c1 * (1 - t) + c2 * t

            out_a = lerp(s_a, d_a, d_a)
            out_rgb = lerp(s_rgb, d_rgb, d_a)

            self.canvas[y, x, 0:3] = out_rgb
            self.canvas[y, x, 3] = out_a

    def clear(self, color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)):
        """Clear the canvas to a specific color"""
        self.canvas[:, :] = color

    def clear_depth(self, depth: float = 1.0):
        """Clear the depth buffer to a specific value"""
        self.depth[:, :] = depth

    def clear_stencil(self, stencil: int = 0):
        """Clear the stencil buffer to a specific value"""
        self.stencil[:, :] = stencil

    def visualize_ascii(self) -> str:
        """Generate ASCII art visualization of canvas

        Returns:
            String representation of the rasterized image
        """

        # Build ASCII string
        ascii_chars = [" ", "░", "▒", "▓", "█"]  # From light to dark

        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                r, g, b, a = self.canvas[y, x]
                intensity = int(((r + g + b) / 3.0) * (len(ascii_chars) - 1))
                intensity = max(0, min(intensity, len(ascii_chars) - 1))
                line += ascii_chars[intensity]
            lines.append(line)

        return "\n".join(lines)

    def visualize_color_ascii(self) -> str:
        """Generate colorized ASCII visualization (ANSI codes)

        Returns:
            String with ANSI color codes
        """

        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                r, g, b, a = self.canvas[y, x]
                r_ansi = int(r * 255)
                g_ansi = int(g * 255)
                b_ansi = int(b * 255)
                line += f"\x1b[38;2;{r_ansi};{g_ansi};{b_ansi}m█\x1b[0m"
            lines.append(line)

        return "\n".join(lines)

    def generate_ppm_image(self, filepath: str = "rasterizer_output.ppm"):
        """Generate PPM (Portable PixMap) image file

        Args:
            filepath: Output file path
        """

        max_v = 255

        # Write PPM file
        with open(filepath, "w") as f:
            # PPM header
            f.write("P3\n")
            f.write(f"{self.width} {self.height}\n")
            f.write(f"{max_v}\n")

            for y in range(self.height):
                for x in range(self.width):
                    r, g, b, a = self.canvas[y, x]
                    r_i = int(max(0, min(r * max_v, max_v)))
                    g_i = int(max(0, min(g * max_v, max_v)))
                    b_i = int(max(0, min(b * max_v, max_v)))
                    f.write(f"{r_i} {g_i} {b_i} ")
                f.write("\n")

        print(f"Generated PPM image: {filepath}")

    def generate_statistics(self, fragments: List[Fragment]) -> dict:
        """Generate statistics about the rasterized output

        Args:
            fragments: List of fragment objects

        Returns:
            Dictionary with statistics
        """
        if not fragments:
            return {"fragment_count": 0, "coverage": 0.0, "color_ranges": {}}

        color_sum = [0.0, 0.0, 0.0, 0.0]
        min_color = [1.0, 1.0, 1.0, 1.0]
        max_color = [0.0, 0.0, 0.0, 0.0]

        for frag in fragments:
            for i in range(4):
                c = frag.color[i]
                color_sum[i] += c
                min_color[i] = min(min_color[i], c)
                max_color[i] = max(max_color[i], c)

        avg_color = [s / len(fragments) for s in color_sum]

        return {
            "fragment_count": len(fragments),
            "coverage": 100.0 * len(fragments) / (self.width * self.height),
            "average_color": {
                "r": avg_color[0],
                "g": avg_color[1],
                "b": avg_color[2],
                "a": avg_color[3],
            },
            "min_color": {
                "r": min_color[0],
                "g": min_color[1],
                "b": min_color[2],
                "a": min_color[3],
            },
            "max_color": {
                "r": max_color[0],
                "g": max_color[1],
                "b": max_color[2],
                "a": max_color[3],
            },
        }


def print_fragment_summary(fragments: List, visualizer: FragmentVisualizer):
    """Print a comprehensive summary of fragments with visualization

    Args:
        fragments: List of fragment objects
        visualizer: FragmentVisualizer instance
    """
    stats = visualizer.generate_statistics(fragments)

    print("\n" + "=" * 60)
    print("RASTERIZER OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Fragment Count: {stats['fragment_count']}")
    print(f"Coverage: {stats['coverage']:.2f}%")
    print("\nAverage Color (RGBA):")
    for component, value in stats["average_color"].items():
        print(f"  {component.upper()}: {value:.4f}")
    print("\nColor Ranges:")
    print(f"  R: {stats['min_color']['r']:.4f} -> {stats['max_color']['r']:.4f}")
    print(f"  G: {stats['min_color']['g']:.4f} -> {stats['max_color']['g']:.4f}")
    print(f"  B: {stats['min_color']['b']:.4f} -> {stats['max_color']['b']:.4f}")
    print(f"  A: {stats['min_color']['a']:.4f} -> {stats['max_color']['a']:.4f}")

    print("\nASCII Visualization (Intensity-based):")
    print("(█ = high intensity, ░ = medium, · = low)")
    print("-" * visualizer.width)
    print(visualizer.visualize_ascii(fragments, intensity_char=True))
    print("-" * visualizer.width)
    print("=" * 60 + "\n")
