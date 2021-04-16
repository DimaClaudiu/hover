"""
???+ note "Child classes which are `functionality`-by-`feature` products."
    This could resemble template specialization in C++.
"""
from .functionality import (
    BokehDataAnnotator,
)
from .feature import BokehForText


class BokehTextAnnotator(BokehDataAnnotator, BokehForText):
    """
    ???+ note "The text flavor of `BokehDataAnnotator`.""
    """

    TOOLTIP_KWARGS = BokehForText.TOOLTIP_KWARGS
    MANDATORY_COLUMNS = BokehForText.MANDATORY_COLUMNS
    SUBSET_GLYPH_KWARGS = BokehDataAnnotator.SUBSET_GLYPH_KWARGS

    def _layout_widgets(self):
        """Define the layout of widgets."""
        from bokeh.layouts import column, row

        layout_rows = (
            row(self.search_pos, self.search_neg),
            # row(self.data_key_button_group),
            row(self.annotator_input, self.annotator_apply, self.annotator_export),
        )
        return column(*layout_rows)

