from __future__ import annotations

import unittest
from dataclasses import replace

from rose.selector import CheckpointDiagnostics, is_valid_checkpoint, select_checkpoint


def row(epoch: int, loss: float) -> CheckpointDiagnostics:
    return CheckpointDiagnostics(
        epoch=epoch,
        loss_spec=loss,
        graph_partition_nmi=0.95,
        active_clusters=3,
        cluster_entropy=0.95,
        largest_cluster_fraction=0.40,
        effective_rank_ratio=0.90,
        feature_variance_mean=0.10,
        strict_observed_invariant=True,
    )


class SelectorTests(unittest.TestCase):
    def test_minimum_loss_with_earlier_tie(self) -> None:
        selected = select_checkpoint(
            [row(20, 0.3), row(40, 0.2), row(60, 0.2)],
            3,
            topology_ceiling=False,
        )
        self.assertEqual(selected.epoch, 40)

    def test_warmup_is_enforced(self) -> None:
        selected = select_checkpoint(
            [row(0, 0.01), row(20, 0.2)],
            3,
            topology_ceiling=False,
        )
        self.assertEqual(selected.epoch, 20)

    def test_fallback_can_select_recorded_epoch_zero(self) -> None:
        collapsed_early = replace(row(0, 0.01), active_clusters=1)
        collapsed_late = replace(row(20, 0.20), active_clusters=1)
        selected = select_checkpoint(
            [collapsed_early, collapsed_late],
            3,
            topology_ceiling=False,
        )
        self.assertEqual(selected.epoch, 0)

    def test_fallback_works_when_all_records_precede_minimum_epoch(self) -> None:
        selected = select_checkpoint([row(0, 0.3), row(5, 0.1)], 3,
                                     topology_ceiling=False)
        self.assertEqual(selected.epoch, 5)

    def test_fallback_loss_tie_prefers_earlier_epoch(self) -> None:
        selected = select_checkpoint(
            [replace(row(25, 0.1), active_clusters=1), row(0, 0.1)],
            3, topology_ceiling=True)
        self.assertEqual(selected.epoch, 0)

    def test_valid_candidate_takes_priority_over_lower_loss_fallback(self) -> None:
        selected = select_checkpoint([row(0, 0.001), row(20, 0.2)], 3,
                                     topology_ceiling=True)
        self.assertEqual(selected.epoch, 20)

    def test_empty_trace_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            select_checkpoint([], 3, topology_ceiling=False)

    def test_observed_support_invariant_is_required(self) -> None:
        invalid = replace(row(20, 0.20), strict_observed_invariant=False)
        self.assertFalse(is_valid_checkpoint(invalid, 3))

    def test_weak_connectivity_prefers_earliest_balanced_agreement(self) -> None:
        selected = select_checkpoint(
            [replace(row(20, .4), graph_partition_nmi=.8), row(40, .3), row(60, .2)],
            3, topology_ceiling=True)
        self.assertEqual(selected.epoch, 40)

    def test_late_balance_drift_can_restore_earlier_valid_state(self) -> None:
        selected = select_checkpoint(
            [row(20, .5), row(60, .12),
             replace(row(120, .1), largest_cluster_fraction=.46, cluster_entropy=.94)],
            3, topology_ceiling=False)
        self.assertEqual(selected.epoch, 60)


if __name__ == "__main__":
    unittest.main()
