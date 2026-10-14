import pytest


@pytest.fixture
def assert_public_output():
    def check(value):
        if isinstance(value, dict):
            assert not ({"raw_logit", "raw_logits", "token_id"} & value.keys())
            for item in value.values():
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)
    return check
