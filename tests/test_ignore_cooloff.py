import engine


def test_ignore_cooloff_blocking():
    state = {}
    cooloff = 2
    # initial ignore triggers block
    block = engine.update_ignore_block(state, [engine.AlertEvent("IGNORE", "test")], cooloff, True)
    assert block is True
    # first clean snapshot keeps block
    block = engine.update_ignore_block(state, [], cooloff, True)
    assert block is True
    # second clean snapshot clears block
    block = engine.update_ignore_block(state, [], cooloff, True)
    assert block is False


def test_ignore_not_hard_block():
    state = {}
    block = engine.update_ignore_block(state, [engine.AlertEvent("IGNORE", "test")], 2, False)
    assert block is False
