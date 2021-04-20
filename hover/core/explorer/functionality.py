"""
???+ note "Intermediate classes based on the functionality."
"""
import numpy as np
from bokeh.models import CDSView, IndexFilter
from bokeh.palettes import Category20
from hover import module_config
from hover.utils.misc import current_time
from hover.utils.bokeh_helper import bokeh_hover_tooltip
from .local_config import SOURCE_COLOR_FIELD, SOURCE_ALPHA_FIELD
from .base import BokehBaseExplorer

class BokehDataAnnotator(BokehBaseExplorer):
    """
    ???+ note "Annoate data points via callbacks on the buttons."

        Features:

        - alter values in the 'label' column through the widgets.
    """

    SUBSET_GLYPH_KWARGS = {
        'raw': {
            "constant": {"line_alpha": 0.3, "size":7},
            "search": {
                "size": ("size", 10, 5, 7),
                "alpha": ("alpha", 1, 0, 0.5)
            },
        }
    }

    def _postprocess_sources(self):
        """
        ???+ note "Infer glyph colors from the label dynamically."
        """
        # infer glyph color from labels
        color_dict = self.auto_color_mapping()

        def get_color(label):
            return color_dict.get(label, "red")

        for _key, _df in self.dfs.items():
            _color = _df["label"].apply(get_color).tolist()
            self.sources[_key].add(_color, SOURCE_COLOR_FIELD)

    def _setup_widgets(self):
        """
        ???+ note "Create annotator widgets and assign Python callbacks."
        """
        from bokeh.models import TextInput, Button, Dropdown

        super()._setup_widgets()

        self.annotator_input = TextInput(title="Label:")
        self.annotator_apply = Button(
            label="Apply",
            button_type="primary",
            height_policy="fit",
            width_policy="min",
        )
        self.annotator_export = Dropdown(
            label="Export",
            button_type="warning",
            menu=["parquet", "CSV"],
            height_policy="fit",
            width_policy="min",
        )

        def callback_apply():
            """
            A callback on clicking the 'self.annotator_apply' button.

            Update labels in the source.
            """
            label = self.annotator_input.value
            selected_idx = self.sources["raw"].selected.indices
            
            if not selected_idx:
                self._warn(
                    "Attempting annotation: did not select any data points. Eligible subset is 'raw'."
                )
                return
            example_old = self.dfs["raw"].at[selected_idx[0], "label"]
            self.dfs["raw"].at[selected_idx, "label"] = label
            example_new = self.dfs["raw"].at[selected_idx[0], "label"]
            self._good(
                f"Applied {len(selected_idx)} annotations: {label} (e.g. {example_old} -> {example_new})"
            )

            self._update_sources()
            self._good(f"Updated annotator plot at {current_time()}")

        def callback_export(event, path_root=None):
            """
            A callback on clicking the 'self.annotator_export' button.

            Saves the dataframe to a pickle.
            """
            import pandas as pd
            import sys
            
            for i, arg in enumerate(sys.argv):
                if arg == '--args':
                    path_root=sys.argv[i+1]
            path_root = path_root.split('/')[-1]

            export_format = event.item

            # auto-determine the export path root
            timestamp = current_time("%Y%m%d%H%M%S")
            path_root = f"{path_root}_hover-annotated_{timestamp}"

            export_df = pd.concat(self.dfs, axis=0, sort=False, ignore_index=True)

            if export_format == "parquet":
                export_path = f"{path_root}.parquet"
                export_df.to_parquet(export_path)
            elif export_format == "CSV":
                export_path = f"{path_root}.csv"
                export_df.to_csv(export_path, index=False)
            elif export_format == "JSON":
                export_path = f"{path_root}.json"
                export_df.to_json(export_path, orient="records")
            elif export_format == "pickle":
                export_path = f"{path_root}.pkl"
                export_df.to_pickle(export_path)
            else:
                raise ValueError(f"Unexpected export format {export_format}")

            self._good(f"Saved DataFrame to {export_path}")

        # keep the references to the callbacks
        self._callback_apply = callback_apply
        self._callback_export = callback_export

        # assign callbacks
        self.annotator_apply.on_click(self._callback_apply)
        # self.annotator_apply.on_click(self._callback_subset_display)
        self.annotator_export.on_click(self._callback_export)

    def plot(self):
        """
        ???+ note "Re-plot all data points with the new labels."
            Overrides the parent method.
            Determines the label -> color mapping dynamically.
        """
        for _key, _source in self.sources.items():
            self.figure.circle(
                "x",
                "y",
                name=_key,
                color=SOURCE_COLOR_FIELD,
                source=_source,
                **self.glyph_kwargs[_key],
            )
            self._good(f"Plotted subset {_key} with {self.dfs[_key].shape[0]} points")

