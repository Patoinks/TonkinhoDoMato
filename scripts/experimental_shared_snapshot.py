"""Benchmark-only shared transport; slower than plain array snapshots on this host."""
from multiprocessing.shared_memory import SharedMemory
import sys


def read_payload(payload):
    if isinstance(payload, bytes):
        return payload
    name, size = payload
    options = {'track': False} if sys.version_info >= (3, 13) else {}
    shared = SharedMemory(name=name, **options)
    try:
        # Owned bytes ensure deserialized arrays cannot outlive the mapping.
        return bytes(shared.buf[:size])
    finally:
        shared.close()


class SnapshotSlots:
    """One parent-owned buffer per in-flight job; reuse only after completion."""
    def __init__(self, capacity):
        self.buffers = [None] * capacity
        self.free = list(range(capacity))

    def publish(self, payload):
        if not self.free:
            raise RuntimeError('No free actor snapshot slot')
        slot = self.free.pop()
        try:
            shared = self.buffers[slot]
            if shared is None or shared.size < len(payload):
                if shared is not None:
                    shared.close()
                    shared.unlink()
                    self.buffers[slot] = None
                shared = SharedMemory(create=True, size=max(65536, 1 << (len(payload) - 1).bit_length()))
                self.buffers[slot] = shared
            shared.buf[:len(payload)] = payload
            return slot, (shared.name, len(payload))
        except BaseException:
            self.free.append(slot)
            raise

    def release(self, slot):
        if slot in self.free:
            raise RuntimeError('Actor snapshot slot already released')
        self.free.append(slot)

    def close(self):
        for slot, shared in enumerate(self.buffers):
            if shared is not None:
                shared.close()
                shared.unlink()
                self.buffers[slot] = None
