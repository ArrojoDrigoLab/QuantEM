"""The two infrastructure limits that made a real laboratory image unimportable.

Both were invisible until someone dropped a file off an actual microscope.

**A 1 GiB wall.** ``cmd_serve`` called ``waitress.serve`` without
``max_request_body_size``, so waitress's 1 073 741 824-byte default applied. It
is checked against ``Content-Length`` at header time, so waitress answered
``413 Request Entity Too Large ... exceeds max_body of 1073741824`` and closed
the socket while the browser was still streaming -- which reaches the user as
an unexplained network error, with nothing anywhere in the UI mentioning a size
limit. Fourteen of the forty TIFFs over 400 MB in this laboratory's own
collection are larger than that default, several of them by a few megabytes.

**Uploads staged on C:.** Django's ``FILE_UPLOAD_TEMP_DIR`` was unset, so a
``TemporaryUploadedFile`` landed in ``tempfile.gettempdir()`` --
``C:\\Users\\<user>\\AppData\\Local\\Temp`` on the shipped build, because the
Tauri shell exported ``QUANTEM_DATA_DIR`` and never ``TEMP``/``TMP``. A 1 GiB
import therefore wrote ~2 GiB to C: and then copied it across volumes to the
data directory: a violation of the project's "nothing is written to C:" rule
and of the owner's "storage lives with the install" ruling, and a pile of
pointless I/O on top.

The tests below pin the three settings that close both holes, and the fourth
pins the desktop shell to the same temp directory the Python side uses.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from django.conf import settings

REPO_SRC = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_SRC.parent
TAURI_MAIN_RS = PROJECT_ROOT / "desktop" / "src-tauri" / "src" / "main.rs"

#: The largest single image in the reference collection this was sized against,
#: in bytes: a Zeiss Atlas export at 56 890 x 48 638 px, measured 2026-08-10.
#: Nothing about the app should make importing it a special case.
LARGEST_KNOWN_IMAGE_BYTES = 2_074_034_677

#: ``waitress.adjustments.Adjustments.max_request_body_size`` -- the default
#: that used to apply, restated here so this test fails loudly if a waitress
#: upgrade ever changes it under us.
WAITRESS_DEFAULT_MAX_BODY = 1_073_741_824


def _report_from_a_fresh_process(data_dir: Path) -> dict[str, str]:
    """Start Django the way a real launch does and report the upload settings.

    A subprocess, because ``STORAGE_DIR`` is fixed at the first import of
    ``quantem.core.config`` and this process has already done that.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_SRC)
    env["DJANGO_SETTINGS_MODULE"] = "quantem.core.settings"
    env["QUANTEM_DATA_DIR"] = str(data_dir)

    script = textwrap.dedent(
        """
        import json, tempfile
        import django
        django.setup()
        from django.conf import settings
        from quantem.core.config import STORAGE_DIR, TMP_DIR
        print("@@" + json.dumps({
            "storage_dir": str(STORAGE_DIR),
            "tmp_dir": str(TMP_DIR),
            "file_upload_temp_dir": str(settings.FILE_UPLOAD_TEMP_DIR),
            "max_upload_bytes": settings.QUANTEM_MAX_UPLOAD_BYTES,
            "system_temp": tempfile.gettempdir(),
        }))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("@@"))
    return json.loads(line[2:])


# --- Where the bytes of an upload land ------------------------------------


def test_the_upload_temp_dir_resolves_inside_the_data_dir(tmp_path):
    """An upload is staged beside its destination, never in the system temp.

    Two things follow from this and nothing else does: the bytes never touch
    C: on a shipped install, and the temporary file is on the same volume as
    the directory it is about to be moved into, so the move is a rename rather
    than a second full copy of the image.
    """
    data_dir = tmp_path / "data-dir"
    report = _report_from_a_fresh_process(data_dir)

    upload_temp = Path(report["file_upload_temp_dir"])
    assert upload_temp.is_absolute()
    assert upload_temp == Path(report["tmp_dir"])
    assert upload_temp.is_relative_to(Path(report["storage_dir"]))
    assert upload_temp.is_relative_to(data_dir.resolve())
    assert upload_temp.is_dir(), "the directory has to exist before an upload arrives"


def test_the_upload_temp_dir_shares_a_volume_with_where_the_upload_is_going():
    """The regression itself: ``FILE_UPLOAD_TEMP_DIR`` unset means C:.

    Django's default is ``tempfile.gettempdir()``, which on Windows is under
    ``%LOCALAPPDATA%`` -- forbidden by this project, and on a different volume
    from a data directory the user put anywhere else. Different volume means
    the view's move can never be a rename, so every large import pays a second
    full copy of the image no matter how the view is written.

    Asserted against the destination rather than against
    ``tempfile.gettempdir()``, because a running server redirects its own
    process temp directory here too and the comparison would be circular.
    """
    from quantem.core.config import STORAGE_DIR, UPLOADS_DIR

    configured = settings.FILE_UPLOAD_TEMP_DIR
    assert configured, "FILE_UPLOAD_TEMP_DIR must be set, or Django stages on C:"

    resolved = Path(configured).resolve()
    assert resolved.is_relative_to(Path(STORAGE_DIR).resolve())
    assert resolved.drive == Path(UPLOADS_DIR).resolve().drive


# --- How large a request the server will accept ----------------------------


def test_the_body_limit_clears_the_largest_real_image_by_a_wide_margin():
    """The number has to be chosen against real files, not against a round one.

    waitress compares with ``>=``, so a body of exactly the limit is refused
    and the largest *accepted* body is one byte smaller. The margin asserted
    here (eight times the largest image this laboratory has produced) is what
    stops the next microscope upgrade from re-creating the wall.
    """
    limit = settings.QUANTEM_MAX_UPLOAD_BYTES

    assert limit > WAITRESS_DEFAULT_MAX_BODY, "the whole point is to beat waitress's 1 GiB default"
    assert limit - 1 > LARGEST_KNOWN_IMAGE_BYTES
    assert limit >= LARGEST_KNOWN_IMAGE_BYTES * 8
    # Not "unlimited" either: a garbage Content-Length must still be refused
    # before waitress spools it onto the data volume and fills the disk.
    assert limit < 2**48


def test_serve_hands_the_limit_to_waitress():
    """A setting nothing reads is a comment.

    ``cmd_serve`` builds its waitress keyword arguments here so the two that
    matter can be asserted without starting a server.
    """
    from quantem.cli import _waitress_options

    options = _waitress_options()

    assert options["max_request_body_size"] == settings.QUANTEM_MAX_UPLOAD_BYTES
    assert options["threads"] == 8


@pytest.fixture
def readable_oversize_error():
    """Install the patched error class and put waitress back afterwards.

    The patch rebinds a name inside an installed package; leaving it rebound
    would make every later test in the session run against a waitress this
    process quietly edited.
    """
    import waitress.parser as waitress_parser

    from quantem.cli import _install_readable_oversize_error

    original = waitress_parser.RequestEntityTooLarge
    try:
        _install_readable_oversize_error(settings.QUANTEM_MAX_UPLOAD_BYTES)
        yield
    finally:
        waitress_parser.RequestEntityTooLarge = original


def _parse_a_post_declaring(content_length: int, adjustments):
    """Run waitress's own request parser over a header block and return it."""
    from waitress.parser import HTTPRequestParser

    parser = HTTPRequestParser(adjustments)
    head = (
        "POST /api/assets/upload/ HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: multipart/form-data; boundary=abc\r\n"
        f"Content-Length: {content_length}\r\n"
        "\r\n"
    ).encode()
    parser.received(head)
    return parser


def test_waitress_refuses_a_real_lab_image_at_its_own_default():
    """The defect, reproduced in the library's own terms.

    If this ever stops failing, waitress changed its default and the margin
    asserted above needs re-reading rather than trusting.
    """
    from waitress.adjustments import Adjustments
    from waitress.utilities import RequestEntityTooLarge

    parser = _parse_a_post_declaring(LARGEST_KNOWN_IMAGE_BYTES, Adjustments())

    assert Adjustments.max_request_body_size == WAITRESS_DEFAULT_MAX_BODY
    assert isinstance(parser.error, RequestEntityTooLarge)


def test_the_configured_limit_accepts_that_same_image():
    from waitress.adjustments import Adjustments

    adj = Adjustments(max_request_body_size=settings.QUANTEM_MAX_UPLOAD_BYTES)
    parser = _parse_a_post_declaring(LARGEST_KNOWN_IMAGE_BYTES, adj)

    assert parser.error is None


def test_an_oversize_body_is_refused_with_a_sentence_a_human_can_read(
    readable_oversize_error,
):
    """Fast, and in words.

    waitress's own refusal is ``exceeds max_body of 1073741824`` in
    ``text/plain`` -- a number with no unit, a phrase from its internals, and a
    content type the client parses as a failed JSON decode. The replacement is
    the same 413, at the same moment (before a single body byte is read), in
    the ``{"error": ...}`` shape the client already understands.
    """
    from waitress.adjustments import Adjustments

    limit = settings.QUANTEM_MAX_UPLOAD_BYTES
    parser = _parse_a_post_declaring(limit + 1, Adjustments(max_request_body_size=limit))

    assert parser.error is not None
    status, headers, body = parser.error.to_response("QuantEM")

    assert status.startswith("413 ")
    assert ("Content-Type", "application/json") in headers
    payload = json.loads(body)
    assert payload["error_code"] == "upload_too_large"
    assert payload["max_upload_bytes"] == limit

    sentence = payload["error"]
    lowered = sentence.lower()
    assert sentence.endswith(".")
    assert "max_body" not in lowered
    assert "waitress" not in lowered
    assert "content-length" not in lowered
    # Invariant I-12: no user-facing string may contain a shell command. The
    # product is spelled "QuantEM"; a lowercase `quantem ...` is the CLI.
    assert "`" not in sentence
    for command in ("quantem ", "pip install", "python -m"):
        assert command not in sentence
    # It has to say how big the limit is, in a unit a person reads.
    assert " GB" in sentence or " MB" in sentence


def test_the_readable_error_leaves_every_other_waitress_failure_alone(
    readable_oversize_error,
):
    """The patch replaces one error class, not waitress's error handling."""
    from waitress.adjustments import Adjustments
    from waitress.parser import HTTPRequestParser

    parser = HTTPRequestParser(Adjustments())
    parser.received(b"POST /api/assets/upload/ HTTP/1.1\r\nContent-Length: nope\r\n\r\n")

    assert parser.error is not None
    status, headers, _body = parser.error.to_response("QuantEM")
    assert status.startswith("400 ")
    assert ("Content-Type", "text/plain; charset=utf-8") in headers


# --- The desktop shell -----------------------------------------------------


@pytest.mark.skipif(not TAURI_MAIN_RS.is_file(), reason="no desktop shell in this tree")
def test_the_desktop_shell_puts_the_process_temp_dir_inside_the_data_dir():
    """The shipped build's half of "nothing is written to C:".

    ``FILE_UPLOAD_TEMP_DIR`` only moves Django's copy. waitress spools any
    request body over 512 kB through ``tempfile.TemporaryFile`` before Django
    is entered at all, and that reads ``TEMP``/``TMP``. The shell exported
    ``QUANTEM_DATA_DIR`` and the WebView2 profile and stopped there, so the
    first of the two copies still went to C:.

    A source test rather than a Rust test: this repository has no Rust
    toolchain checked out, and what needs pinning is the *agreement* between
    the shell's hard-coded path and :data:`quantem.core.config.TMP_DIR`, which
    only this side can see.
    """
    from quantem.core.config import STORAGE_DIR, TMP_DIR

    source = TAURI_MAIN_RS.read_text(encoding="utf-8")

    for var in ("TEMP", "TMP"):
        assert f'env::set_var("{var}"' in source, (
            f"the shell must export {var}; without it waitress spools the "
            "upload into the user's C: temp directory"
        )

    # The shell has to name the same directory the Python side calls TMP_DIR,
    # or the temp file lands on the right volume but the wrong path -- and a
    # future re-layout of config.py would silently split them.
    relative = TMP_DIR.relative_to(STORAGE_DIR).parts
    expected = "".join(f'.join("{part}")' for part in relative)
    assert expected in source, (
        f"the shell's temp directory must be <data dir>{expected}, matching "
        f"quantem.core.config.TMP_DIR ({TMP_DIR})"
    )
