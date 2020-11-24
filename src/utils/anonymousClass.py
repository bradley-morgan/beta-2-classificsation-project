class Obj:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def compare_attributes(self, other):
        return self.__dict__ == other.__dict__
