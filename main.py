"""Top-level entry point script."""
import warnings as _warnings


def _install_pyannote_warning_suppression() -> None:
    """Install warning filters that silence the three noisy pyannote
    warnings emitted at import time and during inference.

    The reason this lives in ``main.py`` rather than in the diarize
    stage is that two of the three warnings fire *during pyannote's
    import*, and pyannote inserts ~22 of its own filters at the front
    of ``warnings.filters`` at that point, pushing anything we set
    in a stage-level module past the point of effectiveness.  Hooking
    into ``warnings._filters_mutated`` re-prepends our filters at
    index 0 every time the filter list is mutated, so we always win
    the "first match wins" race against pyannote's own filters.
    """
    # Patterns: ``re.match`` anchors at the start of the string.
    # The torchcodec warning starts with a leading newline (an
    # artefact of pyannote's showwarning formatting); the other two
    # warnings start with their prose text directly.
    targets = [
        r"\ntorchcodec is not installed correctly",
        r"TensorFloat-32",
        r"std\(\): degrees of freedom",
        # torchaudio's own future-deprecation warning, emitted when
        # the dubbing pipeline saves a wav via the older backend.
        r"In 2.9, this function's implementation will be changed to use torchaudio",
    ]
    for pat in targets:
        _warnings.filterwarnings(
            "ignore", message=pat, category=UserWarning,
        )

    # Pyannote's import will prepend its own ~22 filters.  We can't
    # undo that from a downstream stage (the import already happened),
    # so we wrap ``_filters_mutated`` to re-prepend our entries every
    # time the list changes.  CPython exposes this as a built-in that
    # is the canonical way to invalidate the warnings filter cache.
    orig = getattr(_warnings, "_filters_mutated", None)
    if orig is None:
        return

    def _wrapped() -> None:
        orig()
        # Move our entries back to index 0 so the "first match wins"
        # rule picks them up.
        ours = []
        for entry in list(_warnings.filters):
            msg = entry[1]
            if msg is not None and any(pat in msg.pattern for pat in targets):
                ours.append(entry)
        for entry in ours:
            try:
                _warnings.filters.remove(entry)
            except ValueError:
                pass
            _warnings.filters.insert(0, entry)

    _warnings._filters_mutated = _wrapped


_install_pyannote_warning_suppression()

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
