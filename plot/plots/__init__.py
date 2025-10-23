"""
Concrete plotting helpers.  Each module exposes a single ``generate_*`` function
that takes a pre-filtered dataframe and renders one figure.
"""

from .bar_plot import generate_bar_plot
from .cut_bar_plot import generate_cut_bar_plot

from .box_plot import BoxplotConfig, generate_boxplot
from .line_plot import generate_line_plot

__all__ = [
    "generate_bar_plot",
    "generate_cut_bar_plot",
    "generate_line_plot",
    "generate_boxplot",
]
