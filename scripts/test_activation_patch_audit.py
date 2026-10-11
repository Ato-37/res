"""Dependency-free boundary/hook tests; does not validate CUDA numerics.

Usage: python -m unittest discover -s scripts -p 'test_activation_patch_audit.py'
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


# Load actual function definitions without importing GPU/model dependencies.
tree = ast.parse(Path(__file__).with_name('activation_patching_sweep.py').read_text())
names = {'patch_bounds', 'patched_sequence_score', 'validate_alignment'}
namespace = {'math': math}
exec(compile(ast.Module(body=[n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]),
     '<audit functions>', 'exec'), namespace)


class Tensor:
    dtype = 'float32'
    device = 'cpu'

    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(values), 1)

    def clone(self):
        return Tensor(self.values)

    def __getitem__(self, key):
        return self.values[key[1]]

    def __setitem__(self, key, value):
        self.values[key[1]] = value


class AuditTests(unittest.TestCase):
    def test_bounds(self):
        bounds = namespace['patch_bounds']
        self.assertEqual(bounds('all', 3, 5), (0, 5))
        self.assertEqual(bounds('prompt', 3, 5), (0, 3))
        self.assertEqual(bounds('answer', 3, 5), (3, 5))
        for args in [('all', 3, 3), ('bad', 3, 5), ('all', 0, 5)]:
            with self.assertRaises(ValueError):
                bounds(*args)

    def run_patch(self, scope, source, fire=True):
        state = {'removed': False}
        def register(hook):
            state['hook'] = hook
            return SimpleNamespace(remove=lambda: state.update(removed=True))
        model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(register_forward_hook=register))])))
        original = Tensor([0] * 5)
        def score(*args):
            if fire:
                output = state['hook'](None, None, (original, 'metadata'))
                self.assertEqual(output[1], 'metadata')
                state['values'] = output[0].values
            return 0.0
        namespace['sequence_score'] = score
        try:
            namespace['patched_sequence_score'](
                model, {'input_ids': Tensor([0] * 3)}, [1, 2], 0, source, scope)
        finally:
            self.assertTrue(state['removed'])
            self.assertEqual(original.values, [0] * 5)
        return state['values']

    def test_preserves_unselected_positions(self):
        source = Tensor([1, 2, 3, 4, 5])
        self.assertEqual(self.run_patch('prompt', source), [1, 2, 3, 0, 0])
        self.assertEqual(self.run_patch('answer', source), [0, 0, 0, 4, 5])
        self.assertEqual(self.run_patch('all', source), source.values)

    def test_bad_shape_and_missing_hook_cleanup(self):
        with self.assertRaises(ValueError):
            self.run_patch('all', Tensor([1]))
        with self.assertRaises(RuntimeError):
            self.run_patch('all', Tensor([1] * 5), fire=False)

    def test_alignment(self):
        namespace['torch'] = SimpleNamespace(is_tensor=lambda x: False)
        check = namespace['validate_alignment']
        check({'input_ids': [1], 'pixel_values_videos': [2]},
              {'input_ids': [1], 'pixel_values_videos': [3]})
        with self.assertRaises(ValueError):
            check({'input_ids': [1]}, {'input_ids': [2]})
        with self.assertRaises(ValueError):
            check({'input_ids': [1]}, {'input_ids': [1], 'video_grid_thw': [1]})


if __name__ == '__main__':
    unittest.main()
