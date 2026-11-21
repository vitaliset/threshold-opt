"""Tests for experiments/analysis/generate_decision_summaries.py."""

import pandas as pd

from experiments.analysis.generate_decision_summaries import (
    compute_winning_family_split,
    format_percent,
    render_family_rates,
)


def test_family_split_keeps_other_winners_and_counts_each_cell_once():
    rows = []
    for dataset, winners in {
        "vector": ["RiskPartitioning", "ThresholdRatioCV"],
        "both": ["LinearScalingCV", "VoronoiTuning"],
        "voronoi": ["VoronoiTuningCV"],
        "neither": ["FullMatrixPartitioningCV", "SequentialSelection"],
    }.items():
        for strategy in winners:
            rows.append(
                {
                    "dataset_name": dataset,
                    "model_name": "model",
                    "optimization_objective_name": "Symmetric Cost",
                    "strategy_name": strategy,
                    "is_winner": True,
                }
            )
    split = compute_winning_family_split(pd.DataFrame(rows))
    assert split["count"].tolist() == [1, 1, 1, 1]
    assert split["denominator"].tolist() == [4] * 4
    assert split["percent"].tolist() == [25.0] * 4


def test_format_percent_uses_whole_numbers_away_from_the_ends():
    assert format_percent(49.77) == "50"
    assert format_percent(40.67) == "41"
    assert format_percent(0.152) == "0.2"
    assert format_percent(1.33) == "1.3"
    assert format_percent(100 * 669 / 670) == "99.9"


def test_render_rounds_each_rate_once():
    split = pd.DataFrame(
        [
            {
                "category": category,
                "count": count,
                "denominator": 525,
                "percent": 100 * count / 525,
            }
            for category, count in (
                ("VectorOnly", 100),
                ("Both", 390),
                ("VoronoiOnly", 7),
                ("Neither", 28),
            )
        ]
    )
    tex = render_family_rates({"max": split})
    assert r"\MulticlassMetricCellCount}{525}" in tex
    assert r"\MulticlassMetricVectorOnly}{19\%}" in tex
    assert r"\MulticlassMetricBoth}{74\%}" in tex
    assert r"\MulticlassMetricVoronoiOnly}{1.3\%}" in tex
    assert r"\MulticlassMetricNeither}{5\%}" in tex
