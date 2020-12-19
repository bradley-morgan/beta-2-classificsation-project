import pandas as pd


class Obj:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def compare_attributes(self, other):
        return self.__dict__ == other.__dict__

    def __call__(self, **kwds):
        self.__dict__.update(kwds)

    def delete(self, item):
        if isinstance(item, list):
            for i in item:
                del self.__dict__[i]
        else:
            del self.__dict__[item]

    def to_dict(self):
        return self.__dict__

    def to_dataframe(self, orient: str, columns=None):
        """
        Converts Anonymous object to a pandas dataframe
        :param orient: Should Object Atrribute names be the 'columns' or the 'index' of the dataframe
        :param columns: 'Only use if orient is 'index', represents the column names
        :return: Dataframe
        """
        return pd.DataFrame.from_dict(self.to_dict(), orient=orient, columns=columns)
