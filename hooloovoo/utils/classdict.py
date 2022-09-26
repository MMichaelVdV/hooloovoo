import re
from typing import List, Optional, Dict


class ClassDict:
    """
    An immutable bidirectional dictionary to map integer class indices to human-readable class labels and vice versa.
    """
    def __init__(self, class_labels: List[str]):
        self._index_to_label: List[str] = class_labels
        self._label_to_index: Dict[str, int] = {label: i for i, label in enumerate(class_labels)}

    def label_of_index(self, index: int, raise_error=True) -> Optional[str]:
        """
        :returns: the human-readable label that belongs to a certain integer class index.
        Returns a ``None`` instead of raising an error if ``raise_error`` is ``False``.
        """
        if not raise_error:
            if len(self._index_to_label) > index:
                return None
        return self._index_to_label[index]

    def index_of_label(self, label: str, raise_error=True) -> Optional[int]:
        """
        :returns: the class index that belongs to a certain human-readable class label
        Returns a ``None`` instead of raising an error if ``raise_error`` is ``False``.
        """
        if raise_error:
            return self._label_to_index[label]
        else:
            return self._label_to_index.get(label)

    def index_that_matches(self, pattern: str):
        """
        :returns: the class index whose label matches the given pattern. An error is raised if there are no matches
        or if there is more than one match. Using this is only recommended for debugging purposes.
        """
        compiled_pattern = re.compile(pattern)
        matches = [(i, label) for i, label in enumerate(self._index_to_label)
                   if compiled_pattern.search(label) is not None]
        if len(matches) > 1:
            matches_report = "\n".join("{:3d}: {}".format(*match) for match in matches)
            raise Exception("multiple matches for pattern {}:\n{}".format(pattern, matches_report))
        elif(len(matches)) < 1:
            raise Exception("no matches for pattern {}".format(pattern))
        else:
            return matches[0][0]

    @property
    def class_labels(self):
        """:returns: a copy of the list of class labels used to construct this ClassDict."""
        return self._index_to_label.copy()
