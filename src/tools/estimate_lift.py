from typing import Optional

import pandas as pd
from scipy import stats


def estimate_lift(
    trends: pd.DataFrame,
    first_n_weeks: Optional[int] = None,
    week_start: Optional[int] = None,
    week_end: Optional[int] = None,
) -> pd.DataFrame:
    """
    Estimates the average lift % over the post-period using a t-interval.

    WHEN TO USE:
        After the human has ratified the control group at the HITL checkpoint.
        Pass the output of check_parallel_trends here.

    WHEN NOT TO USE:
        Before parallel trends have been checked and approved by a human.
        Descriptive questions — use query_metric instead.

    Parameters
    ----------
    trends : pd.DataFrame
        Output of check_parallel_trends. Must contain: week_number, lift_pct, period.
    first_n_weeks : int, optional
        Only use the first N post-period weeks. Overrides week_start/week_end.
    week_start : int, optional
        Start of custom week range (inclusive). Positive integers only.
    week_end : int, optional
        End of custom week range (inclusive). Positive integers only.

    Returns
    -------
    pd.DataFrame
        Single row with columns: lift_pct, ci_lower, ci_upper, n_weeks, significant.
    """
    post = trends[trends["period"] == "post"].copy()

    if first_n_weeks is not None:
        post = post.nsmallest(first_n_weeks, "week_number")
    elif week_start is not None or week_end is not None:
        if week_start is not None:
            post = post[post["week_number"] >= week_start]
        if week_end is not None:
            post = post[post["week_number"] <= week_end]

    if len(post) < 2:
        raise ValueError(
            f"Need at least 2 post-period weeks to compute a confidence interval. "
            f"Got {len(post)}."
        )

    lift_values = post["lift_pct"].values
    n           = len(lift_values)
    mean_lift   = lift_values.mean()
    se          = stats.sem(lift_values)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean_lift, scale=se) if se > 0 else (mean_lift, mean_lift)

    return pd.DataFrame([{
        "lift_pct":    round(mean_lift, 4),
        "ci_lower":    round(ci[0], 4),
        "ci_upper":    round(ci[1], 4),
        "n_weeks":     n,
        "significant": bool(ci[0] > 0 or ci[1] < 0),
    }])
