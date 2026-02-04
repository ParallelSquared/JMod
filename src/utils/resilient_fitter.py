"""
Resilient spectral fitter with hang detection and recovery.

Uses a persistent subprocess to run fits, with automatic kill/respawn
if a fit hangs (e.g., due to cvxopt issues in native code).
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
import queue
from src.logger import logger


def _worker_init(in_q, out_q, fit_func, static_kwargs):
    """Worker process loop - receives spectra, runs fits, returns results."""
    while True:
        try:
            item = in_q.get()
            if item is None:  # Shutdown signal
                break
            idx, dia_spec = item
            result = fit_func(dia_spec, **static_kwargs)
            out_q.put((idx, result, None))
        except Exception as e:
            out_q.put((idx, None, str(e)))


class ResilientFitter:
    """
    Runs spectral fitting in a subprocess with hang detection.

    If a fit takes longer than `timeout` seconds, kills the worker
    and respawns it, then retries the fit once.
    """

    def __init__(self, fit_func, static_kwargs, timeout=30):
        """
        Args:
            fit_func: The fitting function (e.g., fit_to_lib2)
            static_kwargs: Dict of arguments that don't change per spectrum
            timeout: Seconds before considering a fit hung
        """
        self.fit_func = fit_func
        self.static_kwargs = static_kwargs
        self.timeout = timeout
        self.in_q = None
        self.out_q = None
        self.worker = None
        self._spawn_worker()
        self.hangs_recovered = 0

    def _spawn_worker(self):
        """Spawn or respawn the worker process."""
        if self.worker is not None and self.worker.is_alive():
            self.worker.kill()
            self.worker.join(timeout=1)

        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.worker = Process(
            target=_worker_init,
            args=(self.in_q, self.out_q, self.fit_func, self.static_kwargs),
            daemon=True
        )
        self.worker.start()

    def fit(self, dia_spec, idx=0):
        """
        Fit a single spectrum with hang protection.

        Returns:
            The fit result, or empty list if failed after retry.
        """
        self.in_q.put((idx, dia_spec))

        try:
            result_idx, result, error = self.out_q.get(timeout=self.timeout)
            if error:
                logger.warning(f"Fit error on spectrum {idx}: {error}")
                return []
            return result

        except queue.Empty:
            # Hung - kill and respawn
            self.hangs_recovered += 1
            logger.warning(f"Hang detected on spectrum {idx}, respawning worker (recovery #{self.hangs_recovered})")
            self._spawn_worker()

            # Retry once
            self.in_q.put((idx, dia_spec))
            try:
                result_idx, result, error = self.out_q.get(timeout=self.timeout)
                if error:
                    logger.warning(f"Retry error on spectrum {idx}: {error}")
                    return []
                return result
            except queue.Empty:
                logger.error(f"Spectrum {idx} hung twice, skipping")
                self._spawn_worker()
                return []

    def shutdown(self):
        """Clean shutdown of worker process."""
        if self.worker and self.worker.is_alive():
            self.in_q.put(None)  # Shutdown signal
            self.worker.join(timeout=2)
            if self.worker.is_alive():
                self.worker.kill()

    def get_stats(self):
        """Return hang recovery statistics."""
        return {
            'hangs_recovered': self.hangs_recovered
        }


