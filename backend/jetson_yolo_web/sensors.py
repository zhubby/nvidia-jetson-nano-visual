import glob
import os


def read_system_metrics():
    metrics = {
        "temperature_c": _read_temperature(),
        "memory": _read_memory(),
    }
    return metrics


def _read_temperature():
    candidates = glob.glob("/sys/class/thermal/thermal_zone*/temp")
    values = []
    for path in candidates:
        try:
            with open(path, "r") as handle:
                raw = handle.read().strip()
            value = float(raw)
            if value > 1000:
                value = value / 1000.0
            if 0 < value < 130:
                values.append(value)
        except (IOError, ValueError):
            continue
    if not values:
        return None
    return round(max(values), 1)


def _read_memory():
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {
            "total_mb": int(memory.total / 1024 / 1024),
            "used_mb": int(memory.used / 1024 / 1024),
            "percent": round(float(memory.percent), 1),
        }
    except Exception:
        pass

    if not os.path.exists("/proc/meminfo"):
        return None
    values = {}
    try:
        with open("/proc/meminfo", "r") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0])
    except (IOError, ValueError):
        return None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    used = total - available
    return {
        "total_mb": int(total / 1024),
        "used_mb": int(used / 1024),
        "percent": round((used / float(total)) * 100.0, 1),
    }
