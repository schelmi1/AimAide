import time
import ctypes
from ctypes.wintypes import LARGE_INTEGER

def accurate_timing(duration_ms: int) -> float:
    kernel32 = ctypes.windll.kernel32

    INFINITE = 0xFFFFFFFF
    WAIT_FAILED = 0xFFFFFFFF
    CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002

    # Call WaitableTimer w/ CREATE_WAITABLE_TIMER_HIGH_RESOLUTION Flag
    handle = kernel32.CreateWaitableTimerExW(None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, 0x1F0003)
    res = kernel32.SetWaitableTimer(handle, ctypes.byref(LARGE_INTEGER(int(duration_ms * -10000))), 0, None, None, 0,)
    
    start_time = time.perf_counter()
    res = kernel32.WaitForSingleObject(handle, INFINITE)
    kernel32.CancelWaitableTimer(handle)

    return (time.perf_counter() - start_time) * 1000

