"""Dependency-free tests for head-group parsing and scope boundaries."""
import ast
from pathlib import Path
import unittest

tree = ast.parse(Path(__file__).with_name('head_group_activation_patching.py').read_text())
names = {'parse_groups', 'patch_head_bounds', 'parse_layer_spec'}
namespace = {}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), '<head helpers>', 'exec'), namespace)


class HeadGroupTests(unittest.TestCase):
    def test_parse_deduplicates_and_validates(self):
        parse = namespace['parse_groups']
        self.assertEqual(parse('0+1,1+0,all', 3), [[0, 1], list(range(3))])
        self.assertEqual(parse('singletons', 2), [[0], [1]])
        self.assertEqual(parse('adjacent', 3), [[0, 1], [1, 2]])
        self.assertEqual(parse('skip1', 4), [[0, 2], [1, 3]])
        for spec in ('', '3', '0+-1', 'x'):
            with self.assertRaises(ValueError):
                parse(spec, 3)

    def test_scope_bounds(self):
        bounds = namespace['patch_head_bounds']
        self.assertEqual(bounds('all', 4, 7), (0, 7))
        self.assertEqual(bounds('prompt', 4, 7), (0, 4))
        self.assertEqual(bounds('answer', 4, 7), (4, 7))
        with self.assertRaises(ValueError):
            bounds('bad', 4, 7)

    def test_layer_spec_supports_all(self):
        parse = namespace['parse_layer_spec']
        self.assertEqual(parse('all', 3), [0, 1, 2])
        self.assertEqual(parse('0,2', 3), [0, 2])
        for spec in ('', '0,0', '3', 'x'):
            with self.assertRaises(ValueError):
                parse(spec, 3)


if __name__ == '__main__':
    unittest.main()
