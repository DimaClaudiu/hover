"""
???+ note "Building blocks of high-level recipes."

    Includes the following:

    -   functions for creating individual standard explorers appropriate for a dataset.
"""
import hover.core.explorer as hovex


EXPLORER_CATALOG = {
    "annotator": {
        "text": hovex.BokehTextAnnotator,
    }
}


def get_explorer_class(task, feature):
    """
    ???+ note "Get the right `hover.core.explorer` class given a task and a feature."

        Can be useful for dynamically creating explorers without knowing the feature in advance.

        | Param     | Type  | Description                          |
        | :-------- | :---- | :----------------------------------- |
        | `task`    | `str` | name of the task, which can be `"finder"`, `"annotator"`, `"margin"`, `"softlabel"`, or `"snorkel"` |
        | `feature` | `str` | name of the main feature, which can be `"text"`, `"audio"` or `"image"` |

        Usage:
        ```python
        # this creates an instance of BokehTextFinder
        explorer = get_explorer_class("finder", "text")(*args, **kwargs)
        ```
    """
    assert task in EXPLORER_CATALOG, f"Invalid task: {task}"
    assert feature in EXPLORER_CATALOG[task], f"Invalid feature: {feature}"
    return EXPLORER_CATALOG[task][feature]


def standard_annotator(dataset, **kwargs):
    """
    ???+ note "Set up a `BokehDataAnnotator` for a `SupervisableDataset`."

        The annotator has a few standard interactions with the dataset:

        -   read all subsets of the dataset
        -   subscribe to all updates in the dataset
        -   can commit annotations through selections in the "raw" subset

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehDataAnnotator` |
    """
    # auto-detect the (main) feature to use
    # feature = dataset.__class__.FEATURE_KEY
    # explorer_cls = get_explorer_class("annotator", feature)

    # first "static" version of the plot
    subsets = hovex.BokehTextAnnotator.SUBSET_GLYPH_KWARGS.keys()
    annotator = hovex.BokehTextAnnotator.from_dataset(
        dataset,
        {_k: _k for _k in subsets},
        title=" ",
        **kwargs,
    )
    annotator.plot()

    # subscribe for df updates
    dataset.subscribe_update_push(annotator, {_k: _k for _k in subsets})

    # annotators can commit to a dataset
    dataset.subscribe_data_commit(annotator, {"raw": "raw"})

    # annotators by default link the selection for preview
    dataset.subscribe_selection_view(annotator, ["raw", "train", "dev", "test"])
    return annotator
