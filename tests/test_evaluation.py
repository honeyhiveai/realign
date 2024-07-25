import unittest
import sys
import os
import asyncio

# Add the parent directory of 'realign' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.realign.evaluation import Evaluation
from src.realign.datasets import Dataset

class ConcreteEvaluation(Evaluation):
    def __init__(self):
        super().__init__()

    def create_run_data(self, final_state, run_id):
        return {"run_id": run_id, "final_state": final_state}

    async def subroutine(self, run_id, **subroutine_kwargs):
        return f"Subroutine executed for run_id: {run_id}, kwargs: {subroutine_kwargs}"

    def visualization(self):
        # Simulate visualization method
        return "Visualization successful"

class TestEvaluationInitialization(unittest.TestCase):
    def test_initialization(self):
        evaluation_instance = ConcreteEvaluation()
        self.assertIsInstance(evaluation_instance, Evaluation)
        self.assertIsNone(evaluation_instance.dataset)
        self.assertIsInstance(evaluation_instance.evaluators, list)
        self.assertIsInstance(evaluation_instance.run_data, dict)
        self.assertIsInstance(evaluation_instance.eval_results, dict)

class TestEvaluationSubroutine(unittest.TestCase):
    def setUp(self):
        self.evaluation_instance = ConcreteEvaluation()

    def test_subroutine(self):
        asyncio.run(self._test_subroutine())

    async def _test_subroutine(self):
        run_id = 1
        result = await self.evaluation_instance.subroutine(run_id=run_id)
        self.assertIsInstance(result, str)
        self.assertIn(f"run_id: {run_id}", result)

    def test_subroutine_with_kwargs(self):
        asyncio.run(self._test_subroutine_with_kwargs())

    async def _test_subroutine_with_kwargs(self):
        run_id = 2
        kwargs = {"test_arg": "test_value", "another_arg": 42}
        result = await self.evaluation_instance.subroutine(run_id=run_id, **kwargs)
        self.assertIsInstance(result, str)
        self.assertIn(f"run_id: {run_id}", result)
        self.assertIn(f"kwargs: {kwargs}", result)
        self.assertIn("test_arg", result)
        self.assertIn("another_arg", result)
        self.assertIn("test_value", result)
        self.assertIn("42", result)

    def test_subroutine_with_evals(self):
        asyncio.run(self._test_subroutine_with_evals())

    async def _test_subroutine_with_evals(self):
        async def dummy_evaluator(run_data):
            return {"score": 1.0, "result": "Dummy evaluation"}

        self.evaluation_instance.evaluators = [dummy_evaluator]
        run_id = 3

        result = await self.evaluation_instance.subroutine_with_evals(run_id=run_id)
        self.assertIsInstance(result, str)
        self.assertIn(str(run_id), result)
        self.assertIn(run_id, self.evaluation_instance.run_data)
        self.assertIn(run_id, self.evaluation_instance.eval_results)
        self.assertEqual(len(self.evaluation_instance.eval_results[run_id]), 1)
        self.assertEqual(self.evaluation_instance.eval_results[run_id][0]['score'], 1.0)
        self.assertEqual(self.evaluation_instance.eval_results[run_id][0]['result'], "Dummy evaluation")

class TestEvaluationVisualization(unittest.TestCase):
    def test_visualization(self):
        evaluation_instance = ConcreteEvaluation()
        result = evaluation_instance.visualization()
        self.assertEqual(result, "Visualization successful")

if __name__ == '__main__':
    unittest.main()
