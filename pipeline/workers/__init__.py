"""
The live pipeline, split into pieces that run at the same time.

    capture  ->  trigger  ->  judge  ->  database
             queue      queue

Reading a file is forgiving: a four-second Claude call just makes the loop
wait, and the file waits with it. A camera does not wait. It produces frames
at wall-clock rate whether or not anything is ready for them, so a capture
loop that blocks on inference either drops frames or accumulates a latency
gap that grows all shift, and with several cameras it compounds.

So capture never blocks on anything slow. It reads frames, keeps a rolling
buffer of recent ones, and hands the newest to the trigger - dropping rather
than waiting if the trigger has fallen behind, because falling further behind
is the worse failure. The trigger does motion, detection and tracking. The
judge pool calls Claude, which is I/O-bound, so several calls can be in
flight while the trigger keeps working.
"""
from .capture import CaptureWorker          # noqa: F401
from .trigger import TriggerWorker          # noqa: F401
from .judge import JudgeWorker, process_candidate   # noqa: F401
