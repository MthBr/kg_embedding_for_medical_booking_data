
"""
custom_funcs.py, we can put in custom code that gets 
used across more than notebook. 
One example would be downstream data preprocessing 
that is only necessary for a subset of notebooks.
"""


import time

def benchmark(fn):
    def _timing(*a, **kw):
        st = time.perf_counter()
        r = fn(*a, **kw)
        print(f"{fn.__name__} execution: {time.perf_counter() - st} seconds")
        return r

    return _timing











def custom_preprocessor(df):  # give the function a more informative name!!!
    """
    Processes the dataframe such that {insert intent here}. (Write better docstrings than this!!!!)

    Intended to be used under this particular circumstance, with {that other function} called before it, and potentially {yet another function} called after it, but optional.

    :param pd.DataFrame df: A pandas dataframe. Should contain the following columns:
        - col1
        - col2
    :returns: A modified dataframe.
    """
    return (df.groupby('col1').count()['col2'])



