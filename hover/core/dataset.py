"""
???+ note "Dataset classes which extend beyond DataFrames."

    When we supervise a collection of data, these operations need to be simple:

    -   transferring data points between subsets
    -   pulling updates from annotation interfaces
    -   pushing updates to annotation interfaces
    -   getting a 2D embedding
    -   loading data for training models
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from hover import module_config
from hover.core import Loggable
from hover.utils.bokeh_helper import auto_label_color
from hover.utils.misc import current_time
from bokeh.models import (
    Button,
    Dropdown,
    ColumnDataSource,
    DataTable,
    TableColumn,
    HTMLTemplateFormatter,
)
from .local_config import (
    dataset_help_widget,
    COLOR_GLYPH_TEMPLATE,
    DATASET_SUBSET_FIELD,
)


class SupervisableDataset(Loggable):
    """
    ???+ note "Feature-agnostic class for a dataset open to supervision."

        Keeping a DataFrame form and a list-of-dicts ("dictl") form, with the intention that

        - the DataFrame form supports most kinds of operations;
        - the list-of-dicts form could be useful for manipulations outside the scope of pandas;
        - synchronization between the two forms should be called sparingly.
    """

    # 'scratch': intended to be directly editable by other objects, i.e. Explorers
    # labels will be stored but not used for information in hover itself

    FEATURE_KEY = "feature"

    def __init__(
        self,
        raw_dictl,
        feature_key="feature",
        label_key="label",
    ):
        """
        ???+ note "Create (1) dictl and df forms and (2) the mapping between categorical and string labels."
            | Param         | Type   | Description                          |
            | :------------ | :----- | :----------------------------------- |
            | `raw_dictl`   | `list` | list of dicts holding the **to-be-supervised** raw data |
            | `feature_key` | `str`  | the key for the feature in each piece of data |
            | `label_key`   | `str`  | the key for the `**str**` label in supervised data |
        """
        self._info("Initializing...")

        def dictl_transform(dictl, labels=True):
            """
            Burner function to transform the input list of dictionaries into standard format.
            """
            # edge case when dictl is empty or None
            if not dictl:
                return []

            # transform the feature and possibly the label
            key_transform = {feature_key: self.__class__.FEATURE_KEY}
            if labels:
                key_transform[label_key] = "label"

            def burner(d):
                """
                Burner function to transform a single dict.
                """
                if labels:
                    assert label_key in d, f"Expected dict key {label_key}"

                trans_d = {key_transform.get(_k, _k): _v for _k, _v in d.items()}

                if not labels:
                    trans_d["label"] = module_config.ABSTAIN_DECODED

                return trans_d

            return [burner(_d) for _d in dictl]

        self.dictls = {
            "raw": dictl_transform(raw_dictl, labels='label' in raw_dictl[0].keys()),
        }

        self.synchronize_dictl_to_df()
        self.df_deduplicate()
        self.synchronize_df_to_dictl()
        self.setup_widgets()
        # self.setup_label_coding() # redundant if setup_pop_table() immediately calls this again
        self.setup_file_export()
        self.setup_pop_table(width_policy="fit", height_policy="fit")
        self.setup_sel_table(width_policy="fit", width=500, height_policy="max", autosize_mode="none", auto_edit=True, editable=True)
        self._good(f"{self.__class__.__name__}: finished initialization.")

    def copy(self, use_df=True):
        """
        ???+ note "Create another instance, copying over the data entries."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `use_df` | `bool` | whether to use the df or dictl form  |
        """
        if use_df:
            self.synchronize_df_to_dictl()
        return self.__class__(
            raw_dictl=self.dictls["raw"],
            feature_key=self.__class__.FEATURE_KEY,
            label_key="label",
        )

    def to_pandas(self, use_df=True):
        """
        ???+ note "Export to a pandas DataFrame."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `use_df` | `bool` | whether to use the df or dictl form  |
        """
        if not use_df:
            self.synchronize_dictl_to_df()
        dfs = []
        _df = self.dfs['raw'].copy()
        _df[DATASET_SUBSET_FIELD] = 'raw'
        dfs.append(_df)

        return pd.concat(dfs, axis=0)

    @classmethod
    def from_pandas(cls, df, **kwargs):
        """
        ???+ note "Import from a pandas DataFrame."
            | Param    | Type   | Description                          |
            | :------- | :----- | :----------------------------------- |
            | `df` | `DataFrame` | with a "SUBSET" field dividing subsets |
        """
        dictls = {}
        dictls['raw'] = df[df[DATASET_SUBSET_FIELD] == 'raw'].to_dict(
            orient="records"
        )
        return cls(
            raw_dictl=dictls['raw'],
            **kwargs,
        )

    def setup_widgets(self):
        """
        ???+ note "Create `bokeh` widgets for interactive data management."
        """
        self.update_pusher = Button(
            label="Push", button_type="success", height_policy="fit", width_policy="min"
        )
        self.dedup_trigger = Button(
            label="Dedup",
            button_type="warning",
            height_policy="fit",
            width_policy="min",
        )
        self.selection_viewer = Button(
            label="View Selected",
            button_type="primary",
            height_policy="fit",
            width_policy="min",
        )


        self.help_div = dataset_help_widget()

    def view(self):
        """
        ???+ note "Defines the layout of `bokeh` objects when visualized."
        """
        # local import to avoid naming confusion/conflicts
        from bokeh.layouts import row, column

        return column(
            self.help_div,
            row(
                self.selection_viewer,
            ),
            # self.pop_table,
            self.sel_table,
        )

    def subscribe_update_push(self, explorer, subset_mapping):
        """
        ???+ note "Enable pushing updated DataFrames to explorers that depend on them."
            | Param            | Type   | Description                            |
            | :--------------- | :----- | :------------------------------------- |
            | `explorer`       | `BokehBaseExplorer` | the explorer to register  |
            | `subset_mapping` | `dict` | `dataset` -> `explorer` subset mapping |

            Note: the reason we need this is due to `self.dfs[key] = ...`-like assignments. If DF operations were all in-place, then the explorers could directly access the updates through their `self.dfs` references.
        """
        # local import to avoid import cycles
        from hover.core.explorer.base import BokehBaseExplorer

        assert isinstance(explorer, BokehBaseExplorer)

        def callback_push():
            df_dict = {_v: self.dfs[_k] for _k, _v in subset_mapping.items()}
            explorer._setup_dfs(df_dict)
            explorer._update_sources()

        self.update_pusher.on_click(callback_push)
        self._good(
            f"Subscribed {explorer.__class__.__name__} to dataset pushes: {subset_mapping}"
        )

    def subscribe_selection_view(self, explorer, subsets):
        """
        ???+ note "Enable viewing groups of data entries, specified by a selection in an explorer."
            | Param            | Type   | Description                            |
            | :--------------- | :----- | :------------------------------------- |
            | `explorer`       | `BokehBaseExplorer` | the explorer to register  |
            | `subsets`        | `list` | subset selections to consider          |
        """
        assert (
            isinstance(subsets, list) and len(subsets) > 0
        ), "Expected a non-empty list of subsets"

        def callback_view():
            sel_slices = []
            selected_idx = explorer.sources['raw'].selected.indices
            sub_slice = explorer.dfs['raw'].iloc[selected_idx]
            sel_slices.append(sub_slice)

            selected = pd.concat(sel_slices, axis=0)

            # replace this with an actual display (and analysis)
            self._callback_update_selection(selected)

        self.selection_viewer.on_click(callback_view)
        self._good(
            f"Subscribed {explorer.__class__.__name__} to selection view: {subsets}"
        )

    def setup_label_coding(self, verbose=True, debug=False):
        """
        ???+ note "Auto-determine labels in the dataset, then create encoder/decoder in lexical order."
            Add `"ABSTAIN"` as a no-label placeholder which gets ignored categorically.

            | Param     | Type   | Description                        |
            | :-------- | :----- | :--------------------------------- |
            | `verbose` | `bool` | whether to log verbosely           |
            | `debug`   | `bool` | whether to enable label validation |
        """
        all_labels = set()
        _df = self.dfs['raw']
        _found_labels = set(_df["label"].tolist())
        all_labels = all_labels.union(_found_labels)

        # exclude ABSTAIN from self.classes, but include it in the encoding
        all_labels.discard(module_config.ABSTAIN_DECODED)
        self.classes = sorted(all_labels)
        self.label_encoder = {
            **{_label: _i for _i, _label in enumerate(self.classes)},
            module_config.ABSTAIN_DECODED: module_config.ABSTAIN_ENCODED,
        }
        self.label_decoder = {_v: _k for _k, _v in self.label_encoder.items()}

        if verbose:
            self._good(
                f"Set up label encoder/decoder with {len(self.classes)} classes."
            )
        if debug:
            self.validate_labels()

    def validate_labels(self, raise_exception=True):
        """
        ???+ note "Assert that every label is in the encoder."

            | Param             | Type   | Description                         |
            | :---------------- | :----- | :---------------------------------- |
            | `raise_exception` | `bool` | whether to raise errors when failed |
        """
        _invalid_indices = None
        assert "label" in self.dfs['raw'].columns
        _mask = self.dfs['raw']["label"].apply(lambda x: x in self.label_encoder)
        _invalid_indices = np.where(_mask is False)[0].tolist()
        if _invalid_indices:
            self._fail(f"Subset {'raw'} has invalid labels:")
            self._print({self.dfs['raw'].loc[_invalid_indices]})
            if raise_exception:
                raise ValueError("invalid labels")
            
    def setup_file_export(self):
        self.file_exporter = Dropdown(
            label="Export",
            button_type="warning",
            menu=["parquet", "CSV"],
            height_policy="fit",
            width_policy="min",
        )
        
        def callback_export(event, path_root=None):
            """
            A callback on clicking the 'self.annotator_export' button.
            Saves the dataframe to a pickle.
            """
            import pandas as pd

            export_format = event.item

            # auto-determine the export path root
            if path_root is None:
                timestamp = current_time("%Y%m%d%H%M%S")
                path_root = f"hover-dataset-export-{timestamp}"

            export_df = self.to_pandas(use_df=True)

            if export_format == "parquet":
                export_path = f"{path_root}.parquet"
                export_df.to_parquet(export_path, index=False)
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

        # assign the callback, keeping its reference
        self._callback_export = callback_export
        self.file_exporter.on_click(self._callback_export)
        
    def setup_pop_table(self, **kwargs):
        """
        ???+ note "Set up a bokeh `DataTable` widget for monitoring subset data populations."

            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `**kwargs` |        | forwarded to the `DataTable` |
        """
        pop_source = ColumnDataSource(dict())
        pop_columns = [
            TableColumn(field="label", title="label"),
            *[
                TableColumn(field=f"count_{'raw'}", title='count')
            ],
            TableColumn(
                field="color",
                title="color",
                formatter=HTMLTemplateFormatter(template=COLOR_GLYPH_TEMPLATE),
            ),
        ]
        self.pop_table = DataTable(source=pop_source, columns=pop_columns, **kwargs)

        def update_population():
            """
            Callback function.
            """
            # make sure that the label coding is correct
            self.setup_label_coding()

            # re-compute label population
            eff_labels = [module_config.ABSTAIN_DECODED, *self.classes]
            color_dict = auto_label_color(self.classes)
            eff_colors = [color_dict[_label] for _label in eff_labels]

            pop_data = dict(color=eff_colors, label=eff_labels)
            _subpop = self.dfs['raw']["label"].value_counts()
            pop_data[f"count_{'raw'}"] = [
                _subpop.get(_label, 0) for _label in eff_labels
            ]

            # push results to bokeh data source
            pop_source.data = pop_data

            self._good(
                f"Population updater: latest population with {len(self.classes)} classes."
            )

        update_population()
        self.dedup_trigger.on_click(update_population)

        # store the callback so that it can be referenced by other methods
        self._callback_update_population = update_population

    def setup_sel_table(self, **kwargs):
        """
        ???+ note "Set up a bokeh `DataTable` widget for viewing selected data points."

            | Param      | Type   | Description                  |
            | :--------- | :----- | :--------------------------- |
            | `**kwargs` |        | forwarded to the `DataTable` |
        """
        sel_source = ColumnDataSource(dict())
        sel_columns = [
            TableColumn(
                field="domain",
                title="domain",
                width=125
            ),
               TableColumn(
                field="text",
                title="text",
                width=1000
            )
        ]
        self.sel_table = DataTable(source=sel_source, columns=sel_columns, **kwargs)
        print(sel_columns)

        def update_selection(selected_df):
            """
            Callback function.
            """
            # push results to bokeh data source
            sel_source.data = selected_df.to_dict(orient="list")

            self._good(
                f"Selection updater: latest selection with {selected_df.shape[0]} entries."
            )

        self._callback_update_selection = update_selection

    def df_deduplicate(self):
        """
        ???+ note "Cross-deduplicate data entries by feature between subsets."
        """
        return
        self._info("Deduplicating...")
        # for data entry accounting
        before, after = dict(), dict()

        # deduplicating rule: entries that come LATER are of higher priority

        # keep track of which df has which columns and which rows came from which subset
        columns = dict()
        before['raw'] = self.dfs['raw'].shape[0]
        columns['raw'] = self.dfs['raw'].columns
        self.dfs['raw']["__subset"] = 'raw'

        # concatenate in order and deduplicate
        overall_df = pd.concat(
            [self.dfs['raw']], axis=0, sort=False
        )
        overall_df.drop_duplicates(
            subset=[self.__class__.FEATURE_KEY], keep="last", inplace=True
        )
        overall_df.reset_index(drop=True, inplace=True)

        # cut up slices
        self.dfs['raw'] = overall_df[overall_df["__subset"] == 'raw'].reset_index(
            drop=True, inplace=False
        )[columns['raw']]
        after['raw'] = self.dfs['raw'].shape[0]
        self._info(f"--subset {'raw'} rows: {before['raw']} -> {after['raw']}.")

    def synchronize_dictl_to_df(self):
        """
        ???+ note "Re-make dataframes from lists of dictionaries."
        """
        self.dfs = dict()
        _dictl = self.dictls['raw']
        if _dictl:
            _df = pd.DataFrame(_dictl)
            assert self.__class__.FEATURE_KEY in _df.columns
            assert "label" in _df.columns
        else:
            _df = pd.DataFrame(columns=[self.__class__.FEATURE_KEY, "label"])

        self.dfs['raw'] = _df

    def synchronize_df_to_dictl(self):
        """
        ???+ note "Re-make lists of dictionaries from dataframes."
        """
        self.dictls = dict()
        self.dictls['raw'] = self.dfs['raw'].to_dict(orient="records")


class SupervisableTextDataset(SupervisableDataset):
    """
    ???+ note "Can add text-specific methods."
    """

    FEATURE_KEY = "text"
