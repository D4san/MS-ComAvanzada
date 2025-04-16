# utils/timing.py
def elapsed_time(start, end):
    return end - start

def print_time(label, elapsed):
    print(f"{label}: {elapsed:.4f} segundos")
