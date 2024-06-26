"""Module which contains the Coles dataset with a NoSplit splitter."""

from typing import Optional

import torch
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import NoSplit


class NoSplitDataset(ColesDataset):
    """Custom coles dataset inhereted from ptls coles dataset."""

    def __init__(
        self,
        data: list[dict[str, torch.Tensor]],
        *args,
        min_len: Optional[int] = None,
        max_len: Optional[int] = None,
        deterministic: Optional[bool] = None,
        col_time: str = "event_time",
        **kwargs,
    ):
        """Initialize module internal state.

        Args:
        ----
            data (list[dict]): transaction dataframe in the ptls format (list of dicts)
            min_len (int): minimal subsequence length
            max_len (int): maximal subsequence length
            deterministic: should be None, use for consistency
            col_time (str, optional): column name with event time. Defaults to 'event_time'.
            *args: additional arguments to pass to the parent ColesDataset class.
            **kwargs: additional arguments to pass to the parent ColesDataset class.
        """
        super().__init__(
            MemoryMapDataset(
                data, [SeqLenFilter(min_seq_len=min_len, max_seq_len=max_len)]
            ),
            NoSplit(),
            col_time,
            *args,
            **kwargs,
        )
