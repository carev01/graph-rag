"""dpo4 checkpoint selection (spec 2026-10-24-dpo-coverage-pairs-design.md section 5).
dpo3's val loss kept falling while the clean replies' log-probability collapsed, so val
loss picked the worst checkpoint. The rule: the last checkpoint whose eval logps/chosen is
within `tolerance` nats of its step-0 value, stopping at the first eval below that line.
build_dpo_nb.py embeds this module verbatim in the notebook: the unit tests and Colab run
the same code."""

LOGPS_KEY = "eval_logps/chosen"


def eval_logps(log_history):
    """(step, eval logps/chosen) per eval, sorted, one per step (a resumed run repeats some)."""
    by_step = {h["step"]: h[LOGPS_KEY] for h in log_history if LOGPS_KEY in h}
    return sorted(by_step.items())


def below_line(start, value, tolerance):
    return value < start - tolerance


def select_step(log_history, tolerance=1.0):
    """The selected step, or None when no eval after step 0 stays within the line."""
    series = eval_logps(log_history)
    if not series or series[0][0] != 0:
        raise ValueError("no step-0 eval (eval_on_start) in the log history")
    start = series[0][1]
    chosen = None
    for step, value in series[1:]:
        if below_line(start, value, tolerance):
            break
        chosen = step
    return chosen
