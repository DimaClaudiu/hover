"""
???+ note "Useful subroutines for working with bokeh in general."
"""
import warnings
from functools import wraps
from traceback import format_exc
from bokeh.models import PreText
from bokeh.layouts import column
from bokeh.palettes import Category10, Category20, Category20b
from hover import module_config


def auto_label_color(labels):
    """
    ???+ note "Create a label->hex color mapping dict."
    """
    use_labels = set(labels)
    use_labels.discard(module_config.ABSTAIN_DECODED)
    use_labels = sorted(use_labels, reverse=False)

    assert len(use_labels) <= 40, "Too many labels to support (max at 40)"
    if len(use_labels) <= 10:
        palette = Category10[10]
    elif len(use_labels) <= 20:
        palette = Category20[20]
    else:
        palette = Category20b[20] + Category20[20]
        
    color_dict = {
        module_config.ABSTAIN_DECODED: "#7a7a7a",  # gainsboro hex code
        **{_l: _c for _l, _c in zip(use_labels, palette)},
    }
    return color_dict


def servable(title=None):
    """
    ???+ note "Create a decorator which returns an app (or "handle" function) to be passed to bokeh."

        Usage:

        First wrap a function that creates bokeh plot elements:

        ```python
        @servable()
        def dummy(*args, **kwargs):
            from hover.core.explorer import BokehCorpusAnnotator
            annotator = BokehCorpusAnnotator(*args, **kwargs)
            annotator.plot()

            return annotator.view()
        ```

        Then serve the app in your preferred setting:

        === "inline"
            ```python
            # in a Jupyter cell

            from bokeh.io import show, output_notebook
            output_notebook()
            show(dummy(*args, **kwargs))
            ```

        === "bokeh serve"
            ```python
            # in <your-bokeh-app-dir>/main.py

            from bokeh.io import curdoc
            doc = curdoc()
            dummy(*args, **kwargs)(doc)
            ```

        === "embedded app"
            ```python
            # anywhere in your use case

            from bokeh.server.server import Server
            app_dict = {
                'my-app': dummy(*args, **kwargs),
                'my-other-app': dummy(*args, **kwargs),
            }
            server = Server(app_dict)
            server.start()
            ```
    """

    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            def handle(doc):
                """
                Note that the handle must create a brand new bokeh model every time it is called.

                Reference: https://github.com/bokeh/bokeh/issues/8579
                """
                spinner = PreText(text="loading...")
                layout = column(spinner)

                def progress():
                    spinner.text += "."

                def load():
                    try:
                        bokeh_model = func(*args, **kwargs)
                        # remove spinner and its update
                        try:
                            doc.remove_periodic_callback(progress)
                        except Exception as e:
                            warnings.warn(
                                f"@servable: trying to remove periodic callback, got {type(e)}: {e}"
                            )
                        layout.children.append(bokeh_model)
                        layout.children.pop(0)
                    except Exception as e:
                        # exception handling
                        message = PreText(text=f"{type(e)}: {e}\n{format_exc()}")
                        layout.children.append(message)

                doc.add_root(layout)
                doc.add_periodic_callback(progress, 5000)
                doc.add_timeout_callback(load, 500)
                doc.title = title or func.__name__

            return handle

        return wrapped

    return wrapper


def bokeh_hover_tooltip(
    label=False,
    text=False,
    image=False,
    audio=False,
    coords=True,
    index=True,
    custom=None,
):
    """
    ???+ note "Create a Bokeh hover tooltip from a template."

        - param label: whether to expect and show a "label" field.
        - param text: whether to expect and show a "text" field.
        - param image: whether to expect and show an "image" (url/path) field.
        - param audio: whether to expect and show an "audio" (url/path) field.
        - param coords: whether to show xy-coordinates.
        - param index: whether to show indices in the dataset.
        - param custom: {display: column} mapping of additional (text) tooltips.
    """
    # initialize mutable default value
    custom = custom or dict()

    # prepare encapsulation of a div box and an associated script
    divbox_prefix = """<div class="out tooltip">\n"""
    divbox_suffix = """</div>\n"""
    script_prefix = """<script>\n"""
    script_suffix = """</script>\n"""

    # dynamically add contents to the div box and the script
    divbox = divbox_prefix
    script = script_prefix
    if label:
        divbox += """
        <div>
            <span style="font-size: 16px; color: #966;">
                Label: @label
            </span>
        </div>
        """
    if text:
        divbox += """
        <div style="word-wrap: break-word; width: 95%; text-overflow: ellipsis; line-height: 90%">
            <span style="font-size: 11px;">
                Text: @text
            </span>
        </div>
        """
    if image:
        divbox += """
        <div>
            <span style="font-size: 10px;">
                Image: @image
            </span>
            <img
                src="@image" height="60" alt="@image" width="60"
                style="float: left; margin: 0px 0px 0px 0px;"
                border="2"
            ></img>
        </div>
        """
    if audio:
        divbox += """
        <div>
            <span style="font-size: 10px;">
                Audio: @audio
            </span>
            <audio autoplay preload="auto" src="@audio">
            </audio>
        </div>
        """
    if coords:
        divbox += """
        <div>
            <span style="font-size: 12px; color: #060;">
                Coordinates: ($x, $y)
            </span>
        </div>
        """
    if index:
        divbox += """
        <div>
            <span style="font-size: 12px; color: #066;">
                Index: [$index]
            </span>
        </div>
        """
    for _key, _field in custom.items():
        divbox += f"""
        <div>
            <span style="font-size: 12px; color: #606;">
                {_key}: @{_field}
            </span>
        </div>
        """

    divbox += divbox_suffix
    script += script_suffix
    return divbox + script
