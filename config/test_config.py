import pytest
from config import NestedDict

def test_nested_dict_creation():
    nd = NestedDict(a=1, b=2, c={"x": 10, "y": 20})
    assert nd == {"a": 1, "b": 2, "c": {"x": 10, "y": 20}}
    
def test_nested_dict_modification():
    nd = NestedDict(a=1, b=2, c={"x": 10, "y": 20})
    nd["c"]["z"] = 30
    assert nd == {"a": 1, "b": 2, "c": {"x": 10, "y": 20, "z": 30}}
    
def test_get_all_elements():
    nd = NestedDict(a=1, b=2, c={"x": 10, "y": 20})
    elements = nd.get_all_elements()
    assert elements == {"a": 1, "b": 2, "x": 10, "y": 20}
    
def test_filter_elements_with_func_named_args():
    def filter_func(x, b=5, z='dsfb'):
        return str(b) + z
    
    nd = NestedDict(a=1, b=2, c={"x": 10, "z": 20})
    filtered_elements = nd.filter_elements_with_func_named_args(filter_func)
    assert filtered_elements == {"b": 2, "z": 20}
    
def test_nested_dict_representation():
    nd = NestedDict(a=1, b=2, c={"x": 10, "y": 20})
    repr_string = repr(nd)
    assert repr_string == "NestedDict({'a': 1, 'b': 2, 'c': NestedDict({'x': 10, 'y': 20})})"