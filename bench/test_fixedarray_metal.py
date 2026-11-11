"""Regression tests for the Metal fixedarray_t implementation (issue #1 / P0a).

Covers plain wp.fixedarray usage in kernels on metal:0:
- wp.zeros construction (1D, 2D), indexing, stores, loads, component access
- aug-assign into elements
- passing to @wp.func (typed fixedarray annotation via wp.array)
- forward + backward (tape) through fixedarray element access
- numerical cross-check against the CPU device

Run:  python bench/test_fixedarray_metal.py
Exit code 0 = all passed.
"""

import numpy as np
import warp as wp

DEVICE = "metal:0"
failures = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


@wp.kernel
def fa_construct_index(out: wp.array(dtype=float)):
    i = wp.tid()
    buf = wp.zeros(shape=10, dtype=wp.vec2)
    buf[i] = wp.vec2(float(i), float(i) * 2.0)
    s = 0.0
    for j in range(10):
        s += buf[j][0] + buf[j][1]
    out[i] = s


@wp.kernel
def fa_2d(out: wp.array(dtype=float)):
    i = wp.tid()
    m = wp.zeros(shape=(3, 4), dtype=float)
    m[i, 2] = float(i)
    m[i, 2] += 0.5
    out[i] = m[i, 2]


@wp.kernel
def fa_negative_index(out: wp.array(dtype=float)):
    i = wp.tid()
    buf = wp.zeros(shape=4, dtype=float)
    buf[0] = 1.0
    buf[-1] = 9.0  # aliases buf[3]
    out[i] = buf[3]


@wp.func
def fa_helper(buf: wp.fixedarray(shape=4, dtype=float), k: int) -> float:
    return buf[k] * 2.0


@wp.kernel
def fa_via_func(out: wp.array(dtype=float)):
    i = wp.tid()
    buf = wp.zeros(shape=4, dtype=float)
    buf[i] = float(i) + 1.0
    out[i] = fa_helper(buf, i)


@wp.kernel
def fa_tape_loss(x: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    buf = wp.zeros(shape=4, dtype=wp.float32)
    buf[i] = x[0] * x[0]
    wp.atomic_add(loss, 0, buf[i])


def run_pair(kernel, dim, n_out):
    results = {}
    for dev in ("cpu", DEVICE):
        out = wp.zeros(n_out, dtype=float, device=dev)
        wp.launch(kernel, dim=dim, inputs=[out], device=dev)
        wp.synchronize_device(dev)
        results[dev] = out.numpy().copy()
    return results["cpu"], results[DEVICE]


def main():
    wp.init()
    devs = [str(d) for d in wp.get_devices()]
    if DEVICE not in devs:
        print(f"{DEVICE} not available (devices: {devs}); run under the warp-metal overlay")
        return 1

    cpu, metal = run_pair(fa_construct_index, 4, 4)
    # each thread i: buf[i]=(i,2i), all other entries zero, so the row sum is i+2i = 3i
    expected = np.array([3.0 * i for i in range(4)])
    check("construct+index 1d vec2", np.allclose(metal, expected) and np.allclose(cpu, metal),
          f"metal={metal} expected={expected}")

    cpu, metal = run_pair(fa_2d, 4, 4)
    expected = np.arange(4, dtype=float) + 0.5
    check("2d index + aug-assign", np.allclose(metal, expected) and np.allclose(cpu, metal),
          f"metal={metal}")

    cpu, metal = run_pair(fa_negative_index, 1, 1)
    check("negative index", np.allclose(metal, [9.0]) and np.allclose(cpu, metal), f"metal={metal}")

    cpu, metal = run_pair(fa_via_func, 4, 4)
    expected = 2.0 * (np.arange(4, dtype=float) + 1.0)
    check("fixedarray through @wp.func", np.allclose(metal, expected) and np.allclose(cpu, metal),
          f"metal={metal}")

    # tape through fixedarray
    x = wp.array(np.array([3.0], dtype=np.float32), dtype=wp.float32, device=DEVICE, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=DEVICE, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(fa_tape_loss, dim=1, inputs=[x, loss], device=DEVICE)
    tape.backward(loss=loss)
    g = x.grad.numpy()
    check("tape backward through fixedarray", np.allclose(g, [6.0]), f"grad={g} expected=[6.]")

    print()
    if failures:
        print(f"FAILED: {failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
