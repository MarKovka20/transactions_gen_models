from typing import Optional
import pandas as pd

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin


def time_normalization(x: pd.Series, min_timestamp: int) -> pd.Series:
    """Convert Unix timestmaps to fractions of days, shift times in the dataset.

    Args:
        x (pd.Series) - input datetime column
        min_timestamp (int) - minimum datetime in the dataframe (in Unix timestamp format)

    Returns:
        pd.Series with normalized timestamps (fraction of days)
    """
    return (
        pd.to_datetime(x).astype("datetime64[s]").astype("int64") / 1000000000
        - min_timestamp
    ) / (
        60 * 60 * 24
    )  # seconds in day


class CustomDatetimeNormalization(ColTransformerPandasMixin, ColTransformer):
    """Converts datetime column fraction of days since the earliest transaction in the dataframe.

    Args:
        col_name_original (str) - source column name
        is_drop_original_col (bool) - when target and original columns are different manage original col deletion.
    """

    def __init__(
        self,
        col_name_original: str,
        col_name_target: Optional[str] = None,
        is_drop_original_col: bool = True,
    ) -> None:
        super().__init__(
            col_name_original=col_name_original,
            col_name_target=col_name_target, # type: ignore
            is_drop_original_col=is_drop_original_col
        )
    
    def fit(self, x: pd.DataFrame):
        """Record minimum timestamp
        """
        super().fit(x)
        self.min_timestamp = int(x[self.col_name_original].min().timestamp())

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transform datetime column. Convert unix timestamps into days (float) since 'min_timestamp'."""
        x = self.attach_column(
            x,
            time_normalization(x[self.col_name_original], self.min_timestamp).rename(
                self.col_name_target
            ),
        )
        x = super().transform(x)
        return x