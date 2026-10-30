"""Check matched controls, exact chart values, resume, and real image exports."""
import copy
import csv
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import test_trace_baselines as fixtures
from federated_lora.trace_baselines import run_sessions
from federated_lora.trace_comparison import comparison_configs, run_comparison
from federated_lora.trace_plotting import load_run, plot_comparison, loss_rows, read


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def windows(config, root):
    result = fixtures.manifest()
    condition = config['sessions']['availability']
    result['availability'] = condition
    if condition == 'always':
        for client in result['clients']:
            client['windows'].append({'id':1,'start':result['sessions'][1]['start'],
                'end':result['sessions'][1]['deadline'], 'start_us':result['sessions'][1]['start_us'],
                'end_us':result['sessions'][1]['deadline_us']})
    return result


class ChartRuntime(fixtures.Runtime):
    def train(self, *args):
        adapter, metrics = super().train(*args)
        metrics['batches'] = [{'epoch':0, 'target_tokens':1, 'loss':1.}]
        return adapter, metrics
    def evaluate(self, *args):
        result = super().evaluate(*args)
        result['answer_nll'] = 1.2
        return result


class ChartTests(unittest.TestCase):
    def config(self, algorithm='fedit'):
        cfg = fixtures.TraceBaselineTests().config(algorithm)
        cfg['evaluation']['interval_seconds'] = 20
        return cfg

    def patches(self, stack):
        for module in ('trace_baselines', 'trace_plotting'):
            stack.enter_context(patch(f'federated_lora.{module}.prepare_sessions', side_effect=windows))
        stack.enter_context(patch('federated_lora.scheduler.job_durations', side_effect=fixtures.costs))

    def test_controls_preserve_settings_and_each_original_schedule(self):
        for algo in ('fedit','ffa_lora','flora','fedasync_lora'):
            cfg = self.config(algo)
            original = copy.deepcopy(cfg)
            a, b = list(comparison_configs(cfg))
            self.assertEqual(cfg, original)
            for c in (a, b):
                self.assertEqual(c['federation'], cfg['federation'])
                self.assertEqual(c['training'], cfg['training'])
                self.assertEqual(c['data'], cfg['data'])
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                self.patches(stack)
                stack.enter_context(patch('federated_lora.trace_plotting.render',return_value={}))
                run_comparison(cfg,Path(tmp),runtime_factory=ChartRuntime)
                runs = [load_run(c,Path(tmp)) for c in (a,b)]
                self.assertEqual([len(r['points']) for r in runs],[7,7])
                self.assertGreater(runs[0]['report']['sessions'][1]['server_updates'],0)
                self.assertEqual(runs[1]['report']['sessions'][1]['server_updates'],0)
                for r in runs:
                    self.assertFalse(r['report']['test_used_for_checkpoint_selection'])
                before = [r['report'] for r in runs]
                run_comparison(cfg,Path(tmp),resume=True,runtime_factory=ChartRuntime)
                self.assertEqual([load_run(c,Path(tmp))['report'] for c in (a,b)],before)
                self.assertEqual(ChartRuntime.instances[-1].calls,[])

    def test_rendered_png_svg_and_csv_match_measurements_and_deadlines(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.patches(stack)
            cfg = self.config()
            with patch('federated_lora.trace_plotting.render',return_value={}):
                run_comparison(cfg,Path(tmp),runtime_factory=ChartRuntime)
            result = plot_comparison(cfg,Path(tmp))
            self.assertTrue(result['complete'])
            folder = Path(result['directory'])
            for name in result['plots']:
                self.assertTrue((folder/f'{name}.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                self.assertIn('<svg', (folder/f'{name}.svg').read_text())
            rows = read_csv(folder/'accuracy.csv')
            self.assertEqual(len(rows),14)
            self.assertEqual([r['kind'] for r in rows[:7]],['initial','periodic','periodic','deadline','periodic','periodic','deadline'])
            self.assertAlmostEqual(float(rows[3]['elapsed_session_hours']),60/3600)
            self.assertAlmostEqual(float(rows[6]['elapsed_session_hours']),120/3600)
            self.assertEqual(float(rows[3]['accuracy']),.7)
            deadlines = read_csv(folder/'deadline-comparison.csv')
            self.assertEqual(len(deadlines),4)
            self.assertTrue(all(float(r['gap_to_always_pp'])==0 for r in deadlines))
            self.assertTrue(read_csv(folder/'training-loss.csv'))
            self.assertTrue(read_csv(folder/'batch-loss.csv'))
            # Keep one fixture preview outside the temp folder for visual review.
            preview = Path(__file__).resolve().parents[1]/'exp/chart-tests-preview'
            preview.mkdir(exist_ok=True)
            for p in folder.glob('*.png'):
                (preview/p.name).write_bytes(p.read_bytes())

    def test_reject_wrong_initial_model_and_wrong_score_clock(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.patches(stack)
            stack.enter_context(patch('federated_lora.trace_plotting.render',return_value={}))
            cfg = self.config()
            run_comparison(cfg,Path(tmp),runtime_factory=ChartRuntime)
            a,b = list(comparison_configs(cfg))
            path = Path(tmp)/b['output_dir']/'run_manifest.json'
            identity = read(path)
            identity['initial_adapter_sha256']='different'
            path.write_text(json.dumps(identity))
            with self.assertRaisesRegex(ValueError,'initial adapter'):
                plot_comparison(cfg,Path(tmp))
            path = Path(tmp)/a['output_dir']/'summary.json'
            report = read(path)
            report['evaluations'][0]['elapsed_session_seconds']+=1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'clock'):
                load_run(a,Path(tmp))

    def test_partial_run_excludes_speculative_losses_and_marks_missing_control(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.patches(stack)
            render = stack.enter_context(patch('federated_lora.trace_plotting.render',return_value={}))
            cfg = self.config()
            partial = run_sessions(cfg,Path(tmp),runtime_factory=ChartRuntime,max_events=3)
            self.assertFalse(partial['completed'])
            run = load_run(cfg,Path(tmp))
            self.assertTrue(list((Path(tmp)/cfg['output_dir']/'jobs').glob('*/training.json')))
            self.assertEqual(loss_rows(run),([],[]))
            self.assertEqual(len(run['points']),1)
            self.assertIsNone(render.call_args.args[2].get('always'))

    def test_live_callback_sees_each_saved_evaluation_before_completion(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.patches(stack)
            stack.enter_context(patch('federated_lora.trace_plotting.render',return_value={}))
            cfg = self.config()
            observed = []
            def update():
                report = read(Path(tmp)/cfg['output_dir']/'summary.json')
                observed.append((len(report['evaluations']),report['completed'],report['frozen_base_verified']))
            run_sessions(cfg,Path(tmp),runtime_factory=ChartRuntime,on_report=update)
            self.assertEqual([p[0] for p in observed],[0,1,2,3,4,5,6,6])
            self.assertFalse(observed[0][1])
            self.assertFalse(observed[0][2])
            self.assertEqual(observed[-1],(6,True,True))

    def test_cli_selects_algorithm_and_dry_comparison_never_trains(self):
        from federated_lora.__main__ import main
        import sys
        with patch.object(sys,'argv',['federated_lora','compare','--algorithm','fedit','--simulation-only']),patch('federated_lora.trace_comparison.run_comparison') as call:
            main()
            self.assertEqual(call.call_args.args[0]['federation']['algorithm'],'fedit')
            self.assertTrue(call.call_args.kwargs['simulation_only'])
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            self.patches(stack)
            train = stack.enter_context(patch('federated_lora.trace_baselines.run_sessions'))
            result = run_comparison(self.config(),Path(tmp),simulation_only=True)
            train.assert_not_called()
            self.assertTrue(all(r['accuracy_measured'] is False for r in result.values()))


if __name__=='__main__':
    unittest.main()
