from core.perception.messages import PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady
from core.perception.process_manager import PerceptionProcessManager

def test_process_manager_starts_and_stops_mock():
    m = PerceptionProcessManager()
    m.start(PerceptionWorkerConfiguration('development-perception-mock'))
    assert isinstance(m.get(timeout=10), PerceptionWorkerReady)
    assert m.is_alive
    m.shutdown()
    assert not m.is_alive
