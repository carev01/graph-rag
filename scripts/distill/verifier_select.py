"""Verifier checkpoint selection (spec 2026-10-27-fact-verifier-design.md section 4): the
lowest dev loss, and an early stop once the last `patience` evals all sit above the best
before them (SFT v2 overfitted after about half an epoch). build_verifier_nb.py embeds this
module verbatim in the notebook: the unit tests and Colab run the same code."""


def dev_losses(log_history):
    """(step, eval_loss) per eval, sorted, one per step (a resumed run repeats some)."""
    by_step = {int(h["step"]): float(h["eval_loss"]) for h in log_history if "eval_loss" in h}
    return sorted(by_step.items())


def should_stop(losses, patience):
    return len(losses) > patience and min(losses[-patience:]) > min(losses[:-patience])


def select_dev_step(log_history):
    """The step with the lowest dev loss (the earlier on a tie), or None with no eval."""
    series = dev_losses(log_history)
    return min(series, key=lambda sv: (sv[1], sv[0]))[0] if series else None
