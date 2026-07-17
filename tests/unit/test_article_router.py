from graph_extract.article_router import is_dense_matrix

_DENSE = "intro line\n" + "\n".join("| feat%d | us-east-%d | yes |" % (i, i) for i in range(40))
_PROSE = "AWS Backup centralizes data protection.\n" * 30  # no tables


def test_dense_matrix_true():
    assert is_dense_matrix(_DENSE) is True          # ~40/41 lines are table rows


def test_prose_false():
    assert is_dense_matrix(_PROSE) is False


def test_empty_false():
    assert is_dense_matrix("") is False
    assert is_dense_matrix("   \n  \n") is False


def test_small_table_in_prose_false():
    md = ("AWS Backup notes.\n" * 30) + "\n".join("| a | b |" for _ in range(5))
    assert is_dense_matrix(md) is False              # 5 rows, low ratio, few pipes


def test_pipe_count_triggers_even_if_ratio_low():
    # one very wide table row buried in lots of prose: high pipe count -> dense
    wide = "| " + " | ".join(str(i) for i in range(250)) + " |"
    md = ("prose line\n" * 300) + wide
    assert is_dense_matrix(md) is True               # >=200 pipes


def test_ratio_triggers_even_if_pipes_low():
    # many short table rows: high ratio, modest pipes -> dense
    md = "\n".join("| x |" for _ in range(30)) + "\nprose\nprose"
    assert is_dense_matrix(md) is True               # ratio ~0.9


def test_thresholds_are_configurable():
    md = "\n".join("| x |" for _ in range(3)) + "\nprose\nprose\nprose\nprose\nprose\nprose\nprose"
    # 3/10 = 0.3 ratio; default 0.25 -> dense; raise threshold -> not dense
    assert is_dense_matrix(md) is True
    assert is_dense_matrix(md, ratio_threshold=0.5, pipe_threshold=10_000) is False
