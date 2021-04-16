import hover.core.explorer as hovex


def standard_annotator(dataset, **kwargs):
    """
    ???+ note "Set up a `BokehDataAnnotator` for a `SupervisableDataset`."

        The annotator has a few standard interactions with the dataset:

        -   read all subsets of the dataset
        -   subscribe to all updates in the dataset

        | Param      | Type     | Description                          |
        | :--------- | :------- | :----------------------------------- |
        | `dataset`  | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` | | kwargs to forward to the `BokehDataAnnotator` |
    """
    # auto-detect the (main) feature to use

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


    # annotators by default link the selection for preview
    dataset.subscribe_selection_view(annotator, ["raw"])
    return annotator
