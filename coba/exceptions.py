import sys

with_tb_sys_except_hook = sys.excepthook
sans_tb_sys_except_hook = lambda tp,ex,tb: print(str(ex)) if isinstance(ex, CobaExit) else with_tb_sys_except_hook(tp,ex,tb)

sys.excepthook = sans_tb_sys_except_hook

class CobaException(Exception):
    def _render_traceback_(self):
        # This is a special method used by Jupyter Notebook for writing tracebacks
        # By dummying it up we can prevent CobaException from writing tracebacks in Jupyter Notebook
        # https://ipython.readthedocs.io/en/stable/config/integrating.html
        return [str(self)]

class CobaExit(BaseException):
    # By inheriting directly from BaseException we are able to avoid triggering common 
    # exception handlers. This is how the SystemExit exception also works when calling `exit()`.
    
    def _render_traceback_(self):
        # This is a special method used by Jupyter Notebook for writing tracebacks
        # By dummying it up we can prevent CobaExit from writing tracebacks in Jupyter Notebook
        # https://ipython.readthedocs.io/en/stable/config/integrating.html
        return [str(self)]