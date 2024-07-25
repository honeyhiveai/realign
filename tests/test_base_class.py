import unittest
from realign.base_class import BaseClass
from realign.types import EvalResult, RunData
import os
import json
import asyncio

class ConcreteBaseClass(BaseClass):
    def create_run_data(self, final_state, run_id):
        return {"run_id": run_id, "final_state": final_state}

    def run(self):
        return self

    async def subroutine(self, run_id, **subroutine_kwargs):
        return f"Subroutine executed for run_id: {run_id}"

class TestBaseClassInitialization(unittest.TestCase):
    def test_initialization(self):
        base_instance = ConcreteBaseClass()
        self.assertIsInstance(base_instance, BaseClass)
        self.assertIsNotNone(base_instance.evaluators)
        self.assertIsInstance(base_instance.run_data, dict)
        self.assertIsInstance(base_instance.eval_results, dict)

class TestBaseClassSubroutine(unittest.TestCase):
    async def test_subroutine(self):
        base_instance = ConcreteBaseClass()
        result = await base_instance.subroutine(1)
        self.assertEqual(result, "Subroutine executed for run_id: 1")

    def test_subroutine_wrapper(self):
        asyncio.run(self.test_subroutine())

class TestBaseClassExportEvalResults(unittest.TestCase):
    def test_export_eval_results(self):
        base_instance = ConcreteBaseClass()
        run_data = RunData(final_state="some_state")
        eval_result = EvalResult(score=0.0, result={})
        base_instance.run_data[1] = run_data
        base_instance.eval_results[1] = [eval_result]
        exported_results = base_instance.export_eval_results()
        self.assertIn('run_data_hash', exported_results)
        self.assertIn('metadata', exported_results)
        self.assertIn('evaluations', exported_results)
        self.assertEqual(len(exported_results['run_data_hash']), 1)
        self.assertEqual(len(exported_results['evaluations']), 1)

class TestBaseClassPushEvalsDataset(unittest.TestCase):
    def test_push_evals_dataset(self):
        base_instance = ConcreteBaseClass()
        run_data = RunData(final_state="some_state")
        eval_result = EvalResult(score=0.0, result={})
        base_instance.run_data[1] = run_data
        base_instance.eval_results[1] = [eval_result]
        evaluations_path = 'test_evaluations.json'
        base_instance.push_evals_dataset(evaluations_path)
        self.assertTrue(os.path.exists(evaluations_path))
        with open(evaluations_path, 'r') as f:
            data = json.load(f)
            self.assertIn('run_data_hash', data)
            self.assertIn('metadata', data)
            self.assertIn('evaluations', data)
        os.remove(evaluations_path)

if __name__ == '__main__':
    unittest.main()
