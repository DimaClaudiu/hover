"""
???+ note "High-level functions to produce an interactive annotation interface."
    Stable recipes whose function signatures should almost never change in the future.
"""
from bokeh.layouts import row
from hover.utils.bokeh_helper import servable
from .subroutine import standard_annotator


@servable(title="Simple Annotator")
def simple_annotator(dataset, **kwargs):
    """
    ???+ note "Display the dataset with on a 2D map for annotation."

        | Param     | Type     | Description                          |
        | :-------- | :------- | :----------------------------------- |
        | `dataset` | `SupervisableDataset` | the dataset to link to  |
        | `**kwargs` |       | kwargs to forward to each Bokeh figure |

        Expected visual layout:

        | SupervisableDataset | BokehDataAnnotator |
        | :------------------ | :----------------- |
        | manage data subsets | make annotations   |
    """
    annotator = standard_annotator(dataset, **kwargs)

    sidebar = dataset.view()
    layout = row(sidebar, annotator.view())
    return layout


