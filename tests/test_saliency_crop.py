import pytest
import imaging.saliency_crop as mod

CASES = [
    ("inverse_map_coords", (1,)),
    ("keep_user_result", (1,)),
    ("subject_after_ratio", (1,)),
    ("cache_by_image", (1,)),
    ("align_pixels", (1,)),
    ("trim_black_edge", (1,)),
    ("detect", (1,)),
    ("crop", (1,)),
    ("export", (1,)),
    ("preview", (1,)),
    ("validate_crop", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
