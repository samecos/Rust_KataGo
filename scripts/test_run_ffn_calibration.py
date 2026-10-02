"""CPU failure-boundary tests; never invoke a collector or CUDA."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_ffn_calibration as driver


def fixture():
    inputs = []
    for batch in (1, 3, 8):
        for start in range(0, 2044, batch):
            inputs.append(dict(target_batch=batch, physical_batch=min(batch, 2044-start),
                               tensor_sha256=f'{len(inputs):064x}'))
    plan = dict(mode='complete-calibration', numeric_input_indices=list(range(2044)),
                expected_numeric_rows=2044, expected_forwards=2197, chunk_inputs=64,
                cost=[dict(physical_batch=b, input_indices=list(range(start, start+8)), warmup=3, measurements=4)
                      for b, start in ((1, 0), (3, 2044), (8, 2726))])
    return plan, inputs


class ScheduleTests(unittest.TestCase):
    def test_complete_schedule_and_exact_physical_budget(self):
        plan, inputs = fixture()
        calls = driver.expected_calls(plan, inputs)
        self.assertEqual((len(calls), sum(c['physical_batch'] for c in calls)), (2197, 2656))
        self.assertEqual(sum(c['phase'] == 'cost-warmup' for c in calls), 9)
        self.assertEqual(sum(c['phase'] == 'cost-measure' for c in calls), 96)
        self.assertEqual([c['iteration'] for c in calls[2044:2047]], [0, 1, 2])

    def test_duplicate_missing_or_reordered_numeric_is_rejected(self):
        plan, inputs = fixture()
        for values in (list(range(2043)), [0]+list(range(2043)), list(reversed(range(2044)))):
            with self.subTest(values=values[:3]), self.assertRaises(ValueError):
                driver.expected_calls(dict(plan, numeric_input_indices=values), inputs)

    def test_wrong_warm_measure_or_duplicate_cost_is_rejected(self):
        plan, inputs = fixture()
        for key, value in (('warmup', 4), ('measurements', 3), ('input_indices', [0]*8)):
            broken = copy.deepcopy(plan)
            broken['cost'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                driver.expected_calls(broken, inputs)

    def test_tail_cannot_replace_full_batch(self):
        plan, inputs = fixture()
        for cost_index, tail_index in ((1, 2725), (2, 2981)):
            broken = copy.deepcopy(plan)
            broken['cost'][cost_index]['input_indices'][-1] = tail_index
            with self.subTest(batch=cost_index), self.assertRaisesRegex(ValueError, 'tail'):
                driver.expected_calls(broken, inputs)

    def test_budget_and_negative_index_are_rejected(self):
        plan, inputs = fixture()
        with self.assertRaises(ValueError):
            driver.expected_calls(dict(plan, expected_forwards=2198), inputs)
        plan['cost'][0]['input_indices'][0] = -1
        with self.assertRaises(ValueError):
            driver.expected_calls(plan, inputs)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(projections=[dict(id='g.dual', precision='int8'),
                                       dict(id='g.down', precision='int8'),
                                       dict(id='a.qkv', precision='fp16')])
        self.plan = dict(resolved_recipe_sha256='a'*64, group_ids=['g'])
        self.proposed = dict(recipe=self.recipe, recipe_sha256='a'*64, group_ids=['g'], role='SINGLE_FFN_INT8_ABLATION')
        self.admitted = dict(status='CPU_ACTUAL_GRAPH_ADMISSION_PASS', canonical_recipe_bytes_sha256='a'*64,
            resolved_recipe_sha256='a'*64, group_ids=['g'], role='SINGLE_FFN_INT8_ABLATION', projection_count=3,
            source_object_sha256=driver.sha(driver.canonical(self.recipe, sorted_keys=True)))

    def test_exact_admission_and_role(self):
        self.assertEqual(driver.check_recipe(self.plan, self.recipe, self.proposed, self.admitted),
                         'SINGLE_FFN_INT8_ABLATION')
        for field, value in (('source_object_sha256', 'b'*64), ('role', 'FP16_REFERENCE'), ('projection_count', 4)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                driver.check_recipe(self.plan, self.recipe, self.proposed, dict(self.admitted, **{field:value}))

    def test_attention_quantization_not_a_single_ffn(self):
        self.recipe['projections'][2]['precision'] = 'int8'
        self.admitted['source_object_sha256'] = driver.sha(driver.canonical(self.recipe, sorted_keys=True))
        with self.assertRaisesRegex(ValueError, 'exactly target'):
            driver.check_recipe(self.plan, self.recipe, self.proposed, self.admitted)


class PersistenceTests(unittest.TestCase):
    def test_missing_duplicate_or_budget_changed_registration_is_rejected(self):
        class FakeSources:
            def add(self, source):
                pass
            def json(self, source):
                if source == 'inventory':
                    return []
                return dict(status='PASS', admitted_recipes=80, negative_checks=14, model_parses=2,
                            graph_lowers=2, models=[dict(family='b11'), dict(family='b15')])
        with tempfile.TemporaryDirectory() as tmp:
            reg = dict(schema=driver.SCHEMA, scope='FULL_CALIBRATION_CAPTURE_ONLY', no_retries=True,
                tactics=driver.TACTICS, execution_output=str(Path(tmp)/'capture'), consume_file=str(Path(tmp)/'guard'),
                source_inventory='inventory', executable={}, driver=driver.record(driver.__file__), admission='admission',
                models=[dict(family='b11'),dict(family='b15')], cases=[dict(name=str(n)) for n in range(80)],
                limits=dict(processes=80, uploads=80, forwards=175760, physical_rows=212480, numeric_rows=163520,
                    per_case_timeout_seconds=1800, total_timeout_seconds=43200, per_case_output_bytes=256*1024**2,
                    total_output_bytes=20*1024**3, minimum_free_bytes=25*1024**3))
            for cases in (reg['cases'][:-1], [reg['cases'][0]]*80):
                with self.assertRaisesRegex(ValueError, 'missing/duplicate cases'):
                    driver.validate_registration(dict(reg,cases=cases), FakeSources())
            with self.assertRaisesRegex(ValueError, 'limits differ'):
                driver.validate_registration(dict(reg,limits=dict(reg['limits'],forwards=175761)), FakeSources())

    def test_no_overwrite_keeps_first_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'
            first = driver.put(path, {'state':'first'})
            with self.assertRaises(FileExistsError):
                driver.put(path, {'state':'second'})
            self.assertEqual(driver.record(path), first)
            self.assertTrue(path.with_name(path.name+'.pending').exists())

    def test_source_mutation_is_not_rehashed_as_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'input'
            path.write_bytes(b'abc')
            sources = driver.Sources()
            sources.add(driver.record(path))
            path.write_bytes(b'xyz')
            with self.assertRaises(ValueError):
                sources.recheck()

    def test_driver_consumes_before_failed_launch_and_stops_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exe = root/'fake-exe'
            exe.write_bytes(b'not executed')
            plan = driver.put(root/'plan.json', {'fake':True})
            registration = dict(execution_output=str(root/'capture'), consume_file=str(root/'guard.json'),
                executable=driver.record(exe), limits=dict(minimum_free_bytes=0, total_timeout_seconds=30,
                    per_case_timeout_seconds=10, per_case_output_bytes=100000, total_output_bytes=100000))
            registration_source = driver.put(root/'registration.json', registration)
            cases = [(dict(name=name, plan=plan), {}, []) for name in ('first', 'second')]
            observed = []

            def fail(command, output, name, *args):
                observed.append(name)
                self.assertTrue((root/'guard.json').exists())
                self.assertTrue((output/'first-consumed.json').exists())
                self.assertFalse((output/'second-consumed.json').exists())
                raise ValueError('synthetic launch failure')

            with patch.object(driver, 'validate_registration', return_value=cases), patch.object(driver, 'run_child', side_effect=fail):
                self.assertFalse(driver.execute(root/'registration.json', registration_source['sha256']))
                with self.assertRaisesRegex(ValueError, 'already consumed'):
                    driver.execute(root/'registration.json', registration_source['sha256'])
            self.assertEqual(observed, ['first'])
            result = json.loads((root/'capture/result.json').read_bytes())
            self.assertEqual((result['reserved_cases'], result['verified_cases']), (1, 0))
            self.assertFalse(result['retry_allowed'])


class ChildTests(unittest.TestCase):
    def test_cleanup_error_records_uncertain_exit_without_relaunch(self):
        class FailedCleanup:
            pid = 12345
            returncode = None
            def wait(self, timeout):
                raise subprocess.TimeoutExpired('fake', timeout)
            def poll(self):
                return None
            def kill(self):
                raise OSError('synthetic cleanup failure')
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver.subprocess, 'Popen', return_value=FailedCleanup()) as start:
            with self.assertRaises(ValueError):
                driver.run_child(['fake'], Path(tmp), 'cleanup', os.environ.copy(), .001)
            receipt = json.loads((Path(tmp)/'cleanup-exit.json').read_bytes())
            self.assertIsNone(receipt['actual_exit_code'])
            self.assertEqual(receipt['cleanup_error']['message'], 'synthetic cleanup failure')
            self.assertEqual(start.call_count, 1)

    def test_real_cpu_child_exit_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = driver.run_child([sys.executable, '-c', "print('CPU only')"], Path(tmp), 'ok', os.environ.copy(), 10)
            self.assertEqual(result['actual_exit_code'], 0)
            self.assertIsNone(result['error'])

    def test_failed_spawn_retains_exit_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                driver.run_child([str(root/'missing-executable')], root, 'missing', os.environ.copy(), 1)
            receipt = json.loads((root/'missing-exit.json').read_bytes())
            self.assertIsNone(receipt['pid'])
            self.assertEqual(receipt['error']['type'], 'FileNotFoundError')

    def test_launched_receipt_error_still_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver.put(root/'child-launched.json', {'first':'retained'})
            with self.assertRaises(ValueError):
                driver.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], root, 'child', os.environ.copy(), 10)
            receipt = json.loads((root/'child-exit.json').read_bytes())
            self.assertIsNotNone(receipt['pid'])
            self.assertIsNotNone(receipt['actual_exit_code'])
            self.assertEqual(receipt['error']['type'], 'FileExistsError')

    def test_timeout_kills_owned_child_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            try:
                with self.assertRaisesRegex(ValueError, 'child failed'):
                    driver.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], root, 'timeout', os.environ.copy(), .1)
                receipt = json.loads((root/'timeout-exit.json').read_bytes())
                self.assertIsNotNone(receipt['actual_exit_code'])
                self.assertIsNone(unrelated.poll())
                self.assertLess(receipt['seconds'], 5)
            finally:
                unrelated.kill()
                unrelated.wait(timeout=10)

    def test_monitor_failure_is_terminal_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            def fail():
                raise ValueError('storage limit')
            with self.assertRaises(ValueError):
                driver.run_child([sys.executable, '-c', 'import time; time.sleep(30)'], Path(tmp), 'limit', os.environ.copy(), 10, fail)
            receipt = json.loads((Path(tmp)/'limit-exit.json').read_bytes())
            self.assertEqual(receipt['error']['message'], 'storage limit')
            self.assertIsNotNone(receipt['actual_exit_code'])


if __name__ == '__main__':
    unittest.main()
