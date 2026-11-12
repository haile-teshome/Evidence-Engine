"""Cancel has to stop the work, not just the waiting.

A backend found two days into a session held 52 open connections to Ollama,
still generating answers for requests whose caller had long gone. Aborting the
browser's fetch ends the waiting; it does not reach a pool that was handed the
whole corpus up front. These pin the two halves of the fix: the window is
small, and setting the event stops what has not started.
"""
import threading
import time

import api


def test_runs_everything_when_nobody_cancels():
    done = []
    out, canceled = api.run_cancellable(range(20), lambda i: done.append(i) or i, 4, None)
    assert len(out) == 20
    assert canceled is False


def test_a_cancel_stops_work_that_has_not_started():
    task = "t-cancel-1"
    started = []
    gate = threading.Event()

    def work(i):
        started.append(i)
        gate.wait(timeout=5)          # hold the first workers up
        return i

    def cancel_soon():
        while len(started) < 2:       # wait until the pool is actually busy
            time.sleep(0.01)
        api.cancel_task(api.CancelRequest(task_id=task))
        gate.set()

    threading.Thread(target=cancel_soon, daemon=True).start()
    out, canceled = api.run_cancellable(range(500), work, 2, task)

    assert canceled is True
    # The whole point: 500 items, a window of a handful, so almost none of it
    # ever reaches a worker. The old code submitted all 500 and ran all 500.
    assert len(started) < 20, f"{len(started)} items started despite the cancel"
    assert len(out) <= len(started)


def test_cancel_registration_does_not_leak():
    task = "t-cancel-2"
    api.run_cancellable(range(3), lambda i: i, 2, task)
    # The event is gone, so a later task reusing the id cannot inherit a set one.
    assert api.cancel_task(api.CancelRequest(task_id=task))["canceled"] is False


def test_one_failing_item_does_not_fail_the_batch():
    def work(i):
        if i == 2:
            raise RuntimeError("this item is bad")
        return i

    out, canceled = api.run_cancellable(range(6), work, 3, None)
    assert sorted(out) == [0, 1, 3, 4, 5]
    assert canceled is False


# --- the launcher watchdog ---------------------------------------------------
# The other half of the same story: cancellation cannot help a backend whose
# launcher has gone, because there is nothing left to press Cancel in.


def test_knows_a_live_process_from_a_dead_one():
    import os
    assert api._pid_alive(os.getpid()) is True
    # A pid that has certainly exited: spawn one and reap it.
    import subprocess
    p = subprocess.Popen(["/usr/bin/true"])
    p.wait()
    # Give the OS a moment to clear it, then it must read as gone.
    for _ in range(50):
        if not api._pid_alive(p.pid):
            break
        time.sleep(0.02)
    assert api._pid_alive(p.pid) is False


def test_a_backend_nobody_launched_is_left_alone(monkeypatch):
    """Started by hand, by a test or by a developer: no pid, no watchdog."""
    monkeypatch.delenv("EE_LAUNCHER_PID", raising=False)
    assert api.watch_launcher() is None


def test_a_launched_backend_watches_its_launcher(monkeypatch):
    monkeypatch.setenv("EE_LAUNCHER_PID", "4242")
    t = api.watch_launcher()
    assert t is not None and t.daemon and t.is_alive()
