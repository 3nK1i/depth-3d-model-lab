from core.perception.messages import PerceptionWorkerConfiguration, PerceptionWorkerReady
from core.perception.process_manager import PerceptionProcessManager

def test_mock_worker_can_restart_repeatedly():
    for _ in range(3):
        m = PerceptionProcessManager()
        m.start(PerceptionWorkerConfiguration('development-perception-mock'))
        assert isinstance(m.get(timeout=10), PerceptionWorkerReady)
        m.shutdown()
        assert not m.is_alive
