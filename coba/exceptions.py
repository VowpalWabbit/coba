class CobaException(Exception):
    def _render_traceback_(self):
        #This is a special method used by Jupyter Notebook for writing tracebacks
        #By dummying it up we can prevent CobaException from writing tracebacks in Jupyter Notebook
        pass

class CobaFatal(Exception):
    def _render_traceback_(self):
        #This is a special method used by Jupyter Notebook for writing tracebacks
        #By dummying it up we can prevent CobaException from writing tracebacks in Jupyter Notebook
        pass