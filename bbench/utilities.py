"""Simple one-off utilities with no clear home."""

def check_matplotlib_support(caller_name: str) -> None:
    """Raise ImportError with detailed error message if mpl is not installed.

    Plot utilities like :func:`Results.plot_progressive_stats` should lazily 
    import matplotlib and call this helper before any computation.

    Args:    
        caller_name: The name of the caller that requires matplotlib.

    Remarks:
        This pattern borrows heavily from sklearn. As of 6/20/2020 sklearn code could be found
        at https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/__init__.py
    """
    try:
        import matplotlib
    except ImportError as e:
        raise ImportError(
            f"{caller_name} requires matplotlib."
            " You can install matplotlib with `pip install matplotlib`."
        ) from e