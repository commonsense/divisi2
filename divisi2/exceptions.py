class Error(Exception):
    pass

class LabelError(KeyError):
    pass

class DimensionMismatch(Error):
    pass

