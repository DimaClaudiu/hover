from bokeh.models import Div


DATASET_SUBSET_FIELD = "SUBSET"

COLOR_GLYPH_TEMPLATE = """
<p style="color:<%= value %>;">
    <%= "&#9608;" %>
</p>
"""

DATASET_HELP_HTML = """"""


def dataset_help_widget():
    return Div(text=DATASET_HELP_HTML)
