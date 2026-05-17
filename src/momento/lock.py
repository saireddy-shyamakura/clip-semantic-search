import os
import logging

logger = logging.getLogger(__name__)

class LockFile:
    def __init__(self, path: str):
        self.path = path

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # PID is alive but belongs to another user
            return True

    def acquire(self) -> bool:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    pid_str = f.read().strip()
                if pid_str:
                    old_pid = int(pid_str)
                    if self._pid_alive(old_pid):
                        return False
                    else:
                        logger.warning(f"Removing stale lock file with dead PID {old_pid}")
                        os.remove(self.path)
            except Exception as e:
                logger.error(f"Failed to read/remove lock file: {e}")
                return False

        try:
            with open(self.path, 'w') as f:
                f.write(str(os.getpid()))
            return True
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
            return False

    def release(self) -> None:
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception as e:
                logger.error(f"Failed to remove lock file on release: {e}")
