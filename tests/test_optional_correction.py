"""Correction routing checks with graph, fitting and clustering mocked out."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from rose import pipeline
from rose.correction import NeedDecision


ROOT = Path(__file__).resolve().parents[1]


class OptionalCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = pipeline.load_frozen_method(ROOT / 'config/frozen_method.json')
        self.views = [np.ones((10, 3), np.float32), np.ones((10, 2), np.float32)]
        self.base = np.arange(50, dtype=np.float32).reshape(10, 5) + 1
        self.partition = np.arange(10, dtype=np.int64) % 5

    def _run(self, *, complete: bool, enabled: bool = True,
             admitted: bool = False, threshold: float = .2):
        mask = np.ones((10, 2), np.float32)
        if complete:
            mask[-2:, 0] = 0
        else:
            mask[:5, 1] = 0
            mask[5:, 0] = 0
        settings = replace(self.settings, admission=replace(self.settings.admission, optional=enabled))
        need = NeedDecision(.3, threshold, .25, .35, True, admitted, 1000, .95, 8)
        mapper_result = SimpleNamespace(
            mapper=mock.Mock(name='frozen_mapper'), base_scores=self.base.copy(),
            base_fused=np.ones((10, 32), np.float32), selected=SimpleNamespace(epoch=120))
        graph = SimpleNamespace(adjacency=object(), support={(0, 1)},
                                utilization_alpha=0., utilization_reason='fixture')
        target = SimpleNamespace(target=np.zeros((10, 5), np.float32))
        delta = torch.zeros((10, 5))
        delta[-2:, 0] = .01
        ratio = torch.zeros((10, 1))
        ratio[-2:] = .001
        events = []

        def stage(name, value):
            def call(*args, **kwargs):
                events.append(name)
                return value
            return call

        static = {
            'build_fixed_support_graph': graph,
            'graph_edge_set': {(0, 1)},
            'count_observed_support_violations': 0,
            'build_spectral_target': target,
            'standardize_observed_views': self.views,
            'co_observed_giant_component_ratio': 1.,
            'train_mapper': mapper_result,
            'kmeans_partition': self.partition,
        }
        calls = {}
        with ExitStack() as stack:
            for name, value in static.items():
                calls[name] = stack.enter_context(mock.patch.object(pipeline, name, return_value=value))
            for name, event, value in [
                ('make_counterfactual_batch', 'pairs', {}),
                ('bootstrap_need_decision', 'need', need),
                ('train_corrector', 'fit', SimpleNamespace(corrector=mock.Mock(return_value=torch.zeros((10, 5))))),
                ('bounded_delta', 'project', (delta, ratio)),
            ]:
                calls[name] = stack.enter_context(mock.patch.object(pipeline, name, side_effect=stage(event, value)))
            result = pipeline.run_rose(self.views, mask, 5, 1, settings)
        return result, calls, events, mask, need

    def _assert_skipped(self, result, calls, reason):
        self.assertIsNone(result.need)
        self.assertIsNone(result.correction_training)
        self.assertEqual(result.correction_skip_reason, reason)
        np.testing.assert_array_equal(result.final_scores, self.base)
        np.testing.assert_array_equal(result.correction_delta, np.zeros_like(self.base))
        np.testing.assert_array_equal(result.residual_ratio, np.zeros((10, 1)))
        for name in ['make_counterfactual_batch', 'bootstrap_need_decision', 'train_corrector', 'bounded_delta']:
            calls[name].assert_not_called()
        np.testing.assert_array_equal(calls['kmeans_partition'].call_args.args[0], self.base)

    def test_no_complete_samples_uses_base_without_need_or_corrector(self):
        result, calls, events, _, _ = self._run(complete=False)
        self._assert_skipped(result, calls, 'no_complete_samples')
        self.assertEqual(events, [])

    def test_disabled_correction_skips_need_even_with_complete_samples(self):
        result, calls, _, _, _ = self._run(complete=True, enabled=False)
        self._assert_skipped(result, calls, 'disabled')

    def test_disabled_correction_also_handles_no_complete_samples(self):
        result, calls, _, _, _ = self._run(complete=False, enabled=False)
        self._assert_skipped(result, calls, 'disabled')

    def test_computed_rejection_retains_need_and_base_scores(self):
        result, calls, events, _, need = self._run(complete=True, admitted=False)
        self.assertIs(result.need, need)
        self.assertEqual(result.correction_skip_reason, 'not_admitted')
        self.assertEqual(events, ['pairs', 'need'])
        calls['train_corrector'].assert_not_called()
        calls['bounded_delta'].assert_not_called()
        np.testing.assert_array_equal(result.final_scores, self.base)

    def test_admitted_route_preserves_order_and_arguments(self):
        result, calls, events, _, need = self._run(complete=True, admitted=True)
        self.assertIs(result.need, need)
        self.assertIsNone(result.correction_skip_reason)
        self.assertEqual(events, ['pairs', 'need', 'fit', 'project'])
        self.assertEqual(calls['bootstrap_need_decision'].call_args.kwargs, {
            'seed': 1, 'rho': .1, 'response_budget_factor': 2.,
            'bootstrap_replicates': 1000, 'bootstrap_alpha': 1. - .95,
            'bootstrap_seed': 60013})
        self.assertEqual(calls['train_corrector'].call_args.kwargs['corrector_seed'], 2076)
        self.assertEqual(calls['train_corrector'].call_args.kwargs['settings'], self.settings.correction)
        self.assertEqual(calls['bounded_delta'].call_args.kwargs, {'rho': .1, 'force_complete_identity': True})
        expected = self.base.copy()
        expected[-2:, 0] += .01
        np.testing.assert_array_equal(result.final_scores, expected)

    def test_computed_need_threshold_guard_remains_active(self):
        with self.assertRaisesRegex(RuntimeError, 'threshold differs'):
            self._run(complete=True, threshold=.3)

    def test_cli_serializes_uncomputed_need_as_null(self):
        result, _, _, mask, _ = self._run(complete=False)
        spec = importlib.util.spec_from_file_location('rose_cli_under_test', ROOT / 'run_rose.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = SimpleNamespace(dataset='bdgp', data=Path('unused.mat'), eta=.1,
                               seed=1, output_dir=Path('unused_outputs'), evaluate=False)
        record = {'n_samples': '10', 'n_views': '2', 'n_clusters': '5'}
        mask_record = SimpleNamespace(relative_file='fixture.npz', array_sha256='fixture')
        with ExitStack() as stack:
            for name, value in [('parse_args', args), ('load_data_record', record),
                                ('verify_data_file', 'fixture'),
                                ('load_verified_manifest_mask', (mask, mask_record)),
                                ('load_multi_view_views', self.views), ('run_rose', result)]:
                stack.enter_context(mock.patch.object(cli, name, return_value=value))
            save = stack.enter_context(mock.patch.object(cli.np, 'savez_compressed'))
            labels = stack.enter_context(mock.patch.object(cli, 'load_multi_view_labels', side_effect=AssertionError('No labels in this check')))
            stack.enter_context(mock.patch.object(Path, 'mkdir'))
            written = stack.enter_context(mock.patch.object(Path, 'write_text', return_value=0))
            stack.enter_context(mock.patch('builtins.print'))
            cli.main()
            labels.assert_not_called()
            save.assert_called_once()
            written.assert_called_once()
        summary = json.loads(written.call_args.args[0])
        self.assertIsNone(summary['need'])
        self.assertEqual(summary['correction_skip_reason'], 'no_complete_samples')
        self.assertFalse(summary['correction_applied'])


if __name__ == '__main__':
    unittest.main()
