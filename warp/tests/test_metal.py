# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior specific to the Metal backend, exercised through the public API."""

import gc
import os
import unittest
from typing import Literal

import numpy as np

import warp as wp
import warp.sparse
from warp.tests.unittest_utils import StdOutCapture


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@wp.kernel
def increment_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def scale_kernel(a: wp.array[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.kernel
def saxpy_kernel(x: wp.array[float], y: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = 2.0 * x[i] + y[i]


@wp.struct
class ArrayHolder:
    values: wp.array[float]
    offset: float


@wp.kernel
def holder_sum_kernel(holders: wp.array[ArrayHolder], out: wp.array[float]):
    i = wp.tid()
    h = holders[i]
    out[i] = h.values[0] + h.values[1] + h.offset


@wp.kernel
def print_first_kernel():
    print("metal first string")


@wp.kernel
def print_second_kernel():
    print("metal second string")


@wp.kernel
def mat44_products_twice(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = a[0] * b[0]
    d = a[0] * b[0]
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


@wp.kernel
def mat44_products_chain(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = a[0] * b[0]
    d = (c * b[0]) * (a[0] * c)
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


@wp.kernel
def mat44_products_inverse(a: wp.array[wp.mat44], b: wp.array[wp.mat44], out: wp.array[float]):
    c = wp.inverse(a[0])
    d = (c * a[0]) * b[0]
    for i in range(4):
        for j in range(4):
            out[i * 4 + j] = 2.0 * c[i, j]
            out[16 + i * 4 + j] = 3.0 * d[i, j]


def _make_cholesky_kernel(n: int):
    """Factor and solve one ``n x n`` system per thread block with the tile Cholesky builtins."""

    @wp.kernel(enable_backward=False, module="unique")
    def factor_and_solve(a: wp.array3d[float], b: wp.array2d[float], factor: wp.array3d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n), storage="shared")
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        wp.tile_store(factor[w], t)
        rhs = wp.tile_load(b[w], shape=(n,))
        wp.tile_store(x[w], wp.tile_cholesky_solve(t, rhs, fill_mode="upper"))

    return factor_and_solve


def _make_tile_ops_kernel(n: int):
    """Reductions, scan and sort on one shared ``n``-element tile per thread block."""

    @wp.kernel(enable_backward=False, module="unique")
    def tile_ops(
        a: wp.array2d[float],
        keys_in: wp.array2d[int],
        total: wp.array2d[float],
        lowest: wp.array2d[float],
        highest: wp.array2d[float],
        arg_lowest: wp.array2d[int],
        arg_highest: wp.array2d[int],
        running: wp.array2d[float],
        keys_out: wp.array2d[int],
        order: wp.array2d[int],
    ):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n,), storage="shared")
        wp.tile_store(total[w], wp.tile_sum(t))
        wp.tile_store(lowest[w], wp.tile_min(t))
        wp.tile_store(highest[w], wp.tile_max(t))
        wp.tile_store(arg_lowest[w], wp.tile_argmin(t))
        wp.tile_store(arg_highest[w], wp.tile_argmax(t))
        wp.tile_store(running[w], wp.tile_scan_inclusive(t))
        keys = wp.tile_load(keys_in[w], shape=(n,), storage="shared")
        values = wp.tile_arange(n, dtype=int, storage="shared")
        wp.tile_sort(keys, values)
        wp.tile_store(keys_out[w], keys)
        wp.tile_store(order[w], values)

    return tile_ops


def _make_tile_matmul_kernel(n: int):
    """Compute the returning and the accumulating matrix product of shared ``n x n`` tiles."""

    @wp.kernel(enable_backward=False, module="unique")
    def tile_products(a: wp.array3d[float], b: wp.array3d[float], product: wp.array3d[float], acc: wp.array3d[float]):
        w = wp.tid()
        ta = wp.tile_load(a[w], shape=(n, n), storage="shared")
        tb = wp.tile_load(b[w], shape=(n, n), storage="shared")
        wp.tile_store(product[w], wp.tile_matmul(ta, tb))
        sum_tile = wp.tile_load(a[w], shape=(n, n), storage="shared")
        wp.tile_matmul(ta, tb, sum_tile)
        wp.tile_store(acc[w], sum_tile)

    return tile_products


@wp.kernel(enable_backward=False)
def launch_coords_2d(n1: int, out: wp.array2d[int]):
    i, j = wp.tid()
    row = i * n1 + j
    out[row, 0] = i
    out[row, 1] = j


@wp.kernel(enable_backward=False)
def launch_coords_3d(n1: int, n2: int, out: wp.array2d[int]):
    i, j, k = wp.tid()
    row = (i * n1 + j) * n2 + k
    out[row, 0] = i
    out[row, 1] = j
    out[row, 2] = k


@wp.kernel(enable_backward=False)
def launch_coords_4d(n1: int, n2: int, n3: int, out: wp.array2d[int]):
    i, j, k, l = wp.tid()
    row = ((i * n1 + j) * n2 + k) * n3 + l
    out[row, 0] = i
    out[row, 1] = j
    out[row, 2] = k
    out[row, 3] = l


@wp.kernel(enable_backward=False)
def tiled_launch_coords(block_dim: int, out: wp.array2d[int]):
    i, lane = wp.tid()
    wp.atomic_add(out, i * block_dim + lane, 0, 1)
    out[i * block_dim + lane, 1] = i
    out[i * block_dim + lane, 2] = lane


# Kernels for the Metal fused-Cholesky rewrite (Adjoint._match_metal_fused_cholesky). The rewrite matches source
# shapes, so each case is its own function with literal fill modes rather than one parameterized kernel.
def _fused_upper_register(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
        wp.tile_store(x[w], s)

    return k


def _fused_lower_shared(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t)
        y = wp.tile_load(b[w], shape=(n,), storage="shared")
        s = wp.tile_cholesky_solve(t, y)
        wp.tile_store(x[w], s)

    return k


def _unfused_factor_reused(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
        wp.tile_store(x[w], s)
        total = wp.tile_sum(t)

    return k


def _unfused_offset(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n), offset=(0, 0))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
        wp.tile_store(x[w], s)

    return k


def _unfused_storage(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n), storage="shared")
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
        wp.tile_store(x[w], s)

    return k


def _unfused_fill_modes_differ(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="lower")
        wp.tile_store(x[w], s)

    return k


def _unfused_statement_between(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        t = wp.tile_load(a[w], shape=(n, n))
        wp.tile_cholesky_inplace(t, fill_mode="upper")
        a[w, 0, n - 1] = a[w, 0, n - 1] + 0.0  # a write between the factorization and the solve
        y = wp.tile_load(b[w], shape=(n,))
        s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
        wp.tile_store(x[w], s)

    return k


def _unfused_nested(n):
    @wp.kernel(enable_backward=False, module="unique")
    def k(a: wp.array3d[float], b: wp.array2d[float], x: wp.array2d[float]):
        w = wp.tid()
        if w >= 0:
            t = wp.tile_load(a[w], shape=(n, n))
            wp.tile_cholesky_inplace(t, fill_mode="upper")
            y = wp.tile_load(b[w], shape=(n,))
            s = wp.tile_cholesky_solve(t, y, fill_mode="upper")
            wp.tile_store(x[w], s)

    return k


@wp.kernel(enable_backward=False)
def raise_kernel(code: int):
    wp.metal_raise(code)


# The gathered form of the rewrite, shaped like MuJoCo Warp's _tile_cholesky_factorize_solve_block.
def _gathered(n):
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        start = dof[blk]
        idx = wp.tile_load(elemid, shape=(area,), offset=(blk * area,))
        block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
        L = wp.tile_reshape(block, (n, n))
        wp.tile_cholesky_inplace(L, fill_mode="upper")
        wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
        rhs = wp.tile_load(y[w], shape=(n,), offset=(start,))
        sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
        wp.tile_store(x[w], sol, offset=(start,))

    return k


def _gathered_nested(n):  # the same code inside a block: never rewritten, the reference on Metal
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        if w >= 0:
            start = dof[blk]
            idx = wp.tile_load(elemid, shape=(area,), offset=(blk * area,))
            block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
            L = wp.tile_reshape(block, (n, n))
            wp.tile_cholesky_inplace(L, fill_mode="upper")
            wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
            rhs = wp.tile_load(y[w], shape=(n,), offset=(start,))
            sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
            wp.tile_store(x[w], sol, offset=(start,))

    return k


def _gathered_two_loads(n):
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        start = dof[blk]
        idx = wp.tile_load(elemid, shape=(area,), offset=(blk * area,))
        block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
        L = wp.tile_reshape(block, (n, n))
        wp.tile_cholesky_inplace(L, fill_mode="upper")
        wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
        other = wp.tile_load(y[w], shape=(n,), offset=(start,))
        rhs = wp.tile_load(y[w], shape=(n,), offset=(start,))
        sol = wp.tile_cholesky_solve(L, rhs + other * 0.0, fill_mode="upper")
        wp.tile_store(x[w], sol, offset=(start,))

    return k


def _gathered_load_of_stored_array(n):
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        start = dof[blk]
        idx = wp.tile_load(elemid, shape=(area,), offset=(blk * area,))
        block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
        L = wp.tile_reshape(block, (n, n))
        wp.tile_cholesky_inplace(L, fill_mode="upper")
        wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
        rhs = wp.tile_load(L_out[w], shape=(n,), offset=(adr[start],))  # reads the factor just stored
        sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
        wp.tile_store(x[w], sol, offset=(start,))

    return k


def _gathered_call_in_offset(n):
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        start = dof[blk]
        idx = wp.tile_load(elemid, shape=(area,), offset=(wp.max(blk * area, 0),))
        block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
        L = wp.tile_reshape(block, (n, n))
        wp.tile_cholesky_inplace(L, fill_mode="upper")
        wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
        rhs = wp.tile_load(y[w], shape=(n,), offset=(start,))
        sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
        wp.tile_store(x[w], sol, offset=(start,))

    return k


def _gathered_factor_reused(n):
    area = n * n

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        adr: wp.array[int],
        M: wp.array2d[float],
        elemid: wp.array[int],
        dof: wp.array[int],
        y: wp.array2d[float],
        x: wp.array2d[float],
        L_out: wp.array2d[float],
    ):
        w, blk = wp.tid()
        start = dof[blk]
        idx = wp.tile_load(elemid, shape=(area,), offset=(blk * area,))
        block = wp.tile_load_indexed(M[w], idx, shape=(area,), storage="shared")
        L = wp.tile_reshape(block, (n, n))
        wp.tile_cholesky_inplace(L, fill_mode="upper")
        wp.tile_store(L_out[w], wp.tile_reshape(L, (area,)), offset=(adr[start],))
        rhs = wp.tile_load(y[w], shape=(n,), offset=(start,))
        sol = wp.tile_cholesky_solve(L, rhs, fill_mode="upper")
        wp.tile_store(x[w], sol, offset=(start,))
        total = wp.tile_sum(L)

    return k


def _gathered_problem(n, worlds, rng):
    """Two sparse SPD blocks per world, gathered through an index array with absent and out-of-range entries."""
    pattern = np.triu(rng.random((n, n)) < 0.7, 1)
    pattern = pattern | pattern.T | np.eye(n, dtype=bool)
    a = rng.standard_normal((worlds, n, n)).astype(np.float32) * pattern
    a = (a + a.transpose(0, 2, 1)) * 0.5
    a += (np.abs(a).sum(2).max() + 1.0) * np.eye(n, dtype=np.float32)
    a *= pattern
    slots = int(pattern.sum()) + 7
    position = rng.permutation(slots)
    elemid = np.full(2 * n * n, -1, dtype=np.int32)  # -1: absent pair
    M = np.zeros((worlds, slots), np.float32)
    rows, cols = np.nonzero(pattern)
    elemid[rows * n + cols] = position[: len(rows)]
    M[:, position[: len(rows)]] = a[:, rows, cols]
    absent = np.nonzero(elemid[: n * n] < 0)[0]
    elemid[rng.choice(absent, min(4, len(absent)), replace=False)] = slots + 5  # out of range above
    elemid[n * n :] = elemid[: n * n]
    y = rng.standard_normal((worlds, 2 * n)).astype(np.float32)
    adr = np.zeros(2 * n, dtype=np.int32)
    adr[0], adr[n] = 4, 4 + n * n  # the factors' offsets in L_out, by the block's first dof
    return a, M, elemid, y, adr


@wp.kernel(enable_backward=False)
def strided_view_sum(a: wp.array2d[float], out: wp.array[float]):
    i, j = wp.tid()
    wp.atomic_add(out, i, a[i, j])


@wp.kernel(enable_backward=False)
def indexed_copy(src: wp.indexedarray[float, Literal[2]], dst: wp.array2d[float]):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


@wp.kernel(enable_backward=False)
def world_body_dof_grid(mass: wp.array2d[float], enabled: wp.array2d[int], out: wp.array2d[float]):
    # the shape of MuJoCo Warp's (nworld, nbody, nv) kernels such as _gravity_force: most threads return at once
    w, b, d = wp.tid()
    if enabled[w % enabled.shape[0], b] == 0:
        return
    wp.atomic_add(out[w], d, mass[w % mass.shape[0], b])


def _solver_cholesky(n):  # MuJoCo Warp's _update_gradient_cholesky, line by line
    @wp.func
    def search_sums(g: float, s: float):
        return wp.vec2(g * s, s * s)

    @wp.kernel(enable_backward=False, module="unique")
    def k(
        grad: wp.array2d[float],
        h: wp.array3d[float],
        done: wp.array[bool],
        search: wp.array2d[float],
        dot: wp.array[float],
    ):
        worldid = wp.tid()
        TILE_SIZE = wp.static(n)
        if done[worldid]:
            return
        mat_tile = wp.tile_load(h[worldid], shape=(TILE_SIZE, TILE_SIZE))
        wp.tile_cholesky_inplace(mat_tile, fill_mode="upper")
        input_tile = wp.tile_load(grad[worldid], shape=TILE_SIZE)
        output_tile = wp.tile_cholesky_solve(mat_tile, input_tile, fill_mode="upper")
        sums = wp.tile_reduce(wp.add, wp.tile_map(search_sums, input_tile, output_tile))[0]
        dot[worldid] = sums[0]
        wp.tile_store(search[worldid], wp.tile_map(wp.mul, output_tile, -1.0))

    return k


def _metal_module_source(kernel, block_dim):
    """The Metal source Warp generated for kernel's module, from the kernel cache."""
    import glob  # noqa: PLC0415

    for identifier in (kernel.module.get_module_identifier(block_dim), kernel.module.get_module_identifier()):
        files = glob.glob(os.path.join(wp.config.kernel_cache_dir, identifier, "*.metal"))
        if files:
            with open(files[0]) as f:
                return f.read()
    raise FileNotFoundError(f"no Metal source for {kernel.key} in {wp.config.kernel_cache_dir}")


def _xcrun_metal_available():
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    if shutil.which("xcrun") is None:
        return False
    return subprocess.run(["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True, check=False).returncode == 0


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestMetal(unittest.TestCase):
    device = "metal:0"

    def test_host_memory_prefix_then_whole(self):
        """A host range that starts inside an imported range but extends past it is mapped completely."""
        page = 16384
        n = (3 * page) // 4  # three pages of floats: the prefix import covers only the first page
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        prefix = a[:4]
        wp.launch(increment_kernel, dim=4, inputs=[prefix], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.ones(n, dtype=np.float32)
        expected[:4] = 2.0
        np.testing.assert_array_equal(data, expected)

        # releasing the prefix must not take the mapping of the whole array with it
        del prefix
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        np.testing.assert_array_equal(data, expected + 1.0)

    def test_host_memory_inner_page_then_whole(self):
        """An import in the middle of a larger host array does not shadow the rest of that array."""
        page = 16384
        n = (4 * page) // 4
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        first = page // 4 + 8
        inner = a[first : first + 4]  # lies in the second page only
        wp.launch(increment_kernel, dim=4, inputs=[inner], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        del inner
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.full(n, 2.0, dtype=np.float32)
        expected[first : first + 4] = 3.0
        np.testing.assert_array_equal(data, expected)

    def test_bsr_topology_inside_capture_raises(self):
        """Host-side BSR operations cannot be replayed by a Metal graph, so they refuse to run in a capture."""
        rows = wp.array([0, 1], dtype=int, device=self.device)
        cols = wp.array([0, 1], dtype=int, device=self.device)
        vals = wp.array([1.0, 2.0], dtype=float, device=self.device)
        m = wp.sparse.bsr_zeros(2, 2, block_type=float, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "inside a graph capture"):
            with wp.ScopedCapture(device=self.device):
                wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)  # fine outside a capture
        self.assertEqual(m.nnz_sync(), 2)

    def test_launch_coordinates(self):
        """Multi-dimensional and tiled launches hand every thread its own coordinates, as NumPy unravels them.

        Metal unravels the linear thread index with 32-bit arithmetic; the grids here have odd extents and
        about two million threads, so every division and remainder in the unravel is exercised.
        """
        cases = (
            (launch_coords_2d, (1531, 1409)),
            (launch_coords_3d, (97, 131, 173)),
            (launch_coords_4d, (3, 701, 5, 211)),
        )
        for kernel, shape in cases:
            n = int(np.prod(shape))
            out = wp.full((n, len(shape)), -1, dtype=int, device=self.device)
            wp.launch(kernel, dim=shape, inputs=[*shape[1:], out], device=self.device)
            expected = np.stack(np.unravel_index(np.arange(n), shape), axis=1)
            np.testing.assert_array_equal(out.numpy(), expected, err_msg=f"shape {shape}")

        for block_dim in (32, 64, 256):
            blocks = 4099
            out = wp.zeros((blocks * block_dim, 3), dtype=int, device=self.device)
            wp.launch_tiled(
                tiled_launch_coords, dim=[blocks], inputs=[block_dim, out], device=self.device, block_dim=block_dim
            )
            got = out.numpy()
            np.testing.assert_array_equal(got[:, 0], 1, err_msg=f"block_dim {block_dim}: every thread runs once")
            np.testing.assert_array_equal(got[:, 1], np.repeat(np.arange(blocks), block_dim))
            np.testing.assert_array_equal(got[:, 2], np.tile(np.arange(block_dim), blocks))

    def test_launch_beyond_32_bit_thread_index_raises(self):
        """Metal indexes threads with 32 bits, so a larger grid raises instead of wrapping around."""
        out = wp.zeros((1, 2), dtype=int, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "exceeds the 4294967295 threads"):
            wp.launch(launch_coords_2d, dim=(65536, 65536), inputs=[65536, out], device=self.device)
        with self.assertRaisesRegex(RuntimeError, "exceeds the 4294967295 threads"):
            wp.launch_tiled(
                tiled_launch_coords, dim=[(1 << 32) // 64], inputs=[64, out], device=self.device, block_dim=64
            )

    def _fused_cholesky_case(self, make, n, block_dim, upper, rng):
        """Launch make(n) on Metal: returns (solution, reference, whether the n x n matrix is a shared tile).

        The shared matrix shows in the threadgroup memory the module requests (kept with the cached module too).
        """
        worlds, pad = 3, 2
        m = rng.standard_normal((worlds, n, n)).astype(np.float32)
        a_np = np.full((worlds, n + pad, n + pad), 7.0, dtype=np.float32)  # garbage outside the block
        a_np[:, :n, :n] = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
        b_np = rng.standard_normal((worlds, n)).astype(np.float32)
        x_ref = np.linalg.solve(a_np[:, :n, :n].astype(np.float64), b_np.astype(np.float64)[..., None])[..., 0]
        # only the requested triangle may be read
        rows, cols = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
        a_np[:, rows, cols] = np.nan
        kernel = make(n)
        x = wp.zeros((worlds, n), dtype=float, device=self.device)
        wp.launch_tiled(
            kernel,
            dim=[worlds],
            inputs=[wp.array(a_np, device=self.device), wp.array(b_np, device=self.device), x],
            device=self.device,
            block_dim=block_dim,
        )
        meta = kernel.module.load(wp.get_device(self.device), block_dim).meta
        smem = max(v for key, v in meta.items() if key.endswith("forward_smem_bytes"))
        return x.numpy(), x_ref, smem >= n * n * 4

    def test_fused_cholesky_rewrite(self):
        """The load, factor and solve sequence runs from registers on Metal, without a shared matrix tile.

        Sizes on both sides of the register path's limits (block_dim <= 32, n <= 40): the qualifying ones take
        the fused path, the others compile as written. The triangle the fill mode excludes is poisoned with NaN.
        """
        rng = np.random.default_rng(6)
        for block_dim in (16, 32):
            for n in sorted({block_dim - 1, block_dim, block_dim + 1, 39, 40, 41, 2 * block_dim + 1}):
                for make, upper in ((_fused_upper_register, True), (_fused_lower_shared, False)):
                    msg = f"{make.__name__}, n={n}, block_dim={block_dim}"
                    try:
                        x, x_ref, shared_matrix = self._fused_cholesky_case(make, n, block_dim, upper, rng)
                    except RuntimeError as e:
                        if n <= 41 or "threadgroup memory" not in str(e):
                            raise
                        continue
                    np.testing.assert_allclose(x, x_ref, rtol=1e-3, atol=1e-4, err_msg=msg)
                    self.assertEqual(shared_matrix, n > 40, msg)

    def test_fused_cholesky_rewrite_leaves_other_shapes_alone(self):
        """Code that only resembles the sequence compiles as written and gives the same results as the CPU."""
        rng = np.random.default_rng(7)
        cases = (
            (_unfused_factor_reused, 35, 32),
            (_unfused_offset, 35, 32),
            (_unfused_storage, 35, 32),
            (_unfused_fill_modes_differ, 35, 32),
            (_unfused_statement_between, 35, 32),
            (_unfused_nested, 35, 32),
            (_fused_upper_register, 35, 64),  # block_dim above 32
        )
        for make, n, block_dim in cases:
            msg = f"{make.__name__}, block_dim={block_dim}"
            x, x_ref, shared_matrix = self._fused_cholesky_case(make, n, block_dim, True, np.random.default_rng(8))
            self.assertTrue(shared_matrix, msg)
            if make is _unfused_fill_modes_differ:
                continue  # mixes an upper factor with a lower solve on purpose; compared with the CPU below
            np.testing.assert_allclose(x, x_ref, rtol=1e-3, atol=1e-4, err_msg=msg)

        # an array smaller than the tile: tile_load zero-fills the rest, and so do the fused loads (no read beyond
        # the array); the matrix is then singular, so compare with the unfused CPU code value by value, NaN included
        small = rng.standard_normal((1, 30, 30)).astype(np.float32)
        small = small @ small.transpose(0, 2, 1) + 30 * np.eye(30, dtype=np.float32)
        b_small = rng.standard_normal((1, 35)).astype(np.float32)
        results = []
        for device, block_dim in ((self.device, 32), ("cpu", 1)):
            x = wp.zeros((1, 35), dtype=float, device=device)
            wp.launch_tiled(
                _fused_upper_register(35),
                dim=[1],
                inputs=[wp.array(small, device=device), wp.array(b_small, device=device), x],
                device=device,
                block_dim=block_dim,
            )
            results.append(x.numpy())
        np.testing.assert_array_equal(np.isnan(results[0]), np.isnan(results[1]))
        np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-4, equal_nan=True)

        # the mismatched fill modes compute whatever the unfused code computes, as on the CPU
        a_np = rng.standard_normal((1, 35, 35)).astype(np.float32)
        a_np = a_np @ a_np.transpose(0, 2, 1) + 35 * np.eye(35, dtype=np.float32)
        b_np = rng.standard_normal((1, 35)).astype(np.float32)
        results = []
        for device, block_dim in ((self.device, 32), ("cpu", 1)):
            x = wp.zeros((1, 35), dtype=float, device=device)
            wp.launch_tiled(
                _unfused_fill_modes_differ(35),
                dim=[1],
                inputs=[wp.array(a_np, device=device), wp.array(b_np, device=device), x],
                device=device,
                block_dim=block_dim,
            )
            results.append(x.numpy())
        np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-4)

    def test_device_error_raises_at_synchronize(self):
        """A kernel's wp.metal_raise(code) is raised by the next synchronize, once, and the first error wins."""
        device = wp.get_device(self.device)
        wp.synchronize_device(device)

        wp.launch(raise_kernel, dim=64, inputs=[7], device=device)
        with self.assertRaisesRegex(RuntimeError, "Metal kernel reported error code 7"):
            wp.synchronize_device(device)
        wp.synchronize_device(device)  # cleared

        wp.launch(raise_kernel, dim=1, inputs=[5], device=device)
        wp.launch(raise_kernel, dim=1, inputs=[9], device=device)
        with self.assertRaisesRegex(RuntimeError, "error code 5"):
            wp.synchronize_device(device)
        wp.synchronize_device(device)

        wp.launch(raise_kernel, dim=1, inputs=[1], device=device)
        with self.assertRaisesRegex(RuntimeError, "fused Cholesky: the factor store overlaps"):
            wp.synchronize_device(device)

        # reading an array synchronizes too, so it raises as well
        a = wp.zeros(4, dtype=float, device=device)
        wp.launch(raise_kernel, dim=1, inputs=[11], device=device)
        with self.assertRaisesRegex(RuntimeError, "error code 11"):
            a.numpy()
        np.testing.assert_array_equal(a.numpy(), 0.0)

        # kernels that do not report leave the channel clear
        wp.launch(increment_kernel, dim=4, inputs=[a], device=device)
        wp.synchronize_device(device)

    def test_device_error_raises_after_graph_replay(self):
        """Errors reported by a kernel in a captured graph are raised after the replay, and cleared."""
        device = wp.get_device(self.device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(raise_kernel, dim=8, inputs=[3], device=device)
        wp.synchronize_device(device)  # capturing records, it does not run
        for _ in range(2):
            wp.capture_launch(capture.graph)
            with self.assertRaisesRegex(RuntimeError, "error code 3"):
                wp.synchronize_device(device)
        wp.synchronize_device(device)

    def _gathered_run(self, make, n, device, block_dim, problem, L_out=None):
        a, M, elemid, y, adr = problem
        worlds = a.shape[0]
        x = wp.zeros((worlds, 2 * n), dtype=float, device=device)
        if L_out is None:
            L_out = wp.full((worlds, 2 * n * n + 9), 5.0, dtype=float, device=device)
        kernel = make(n)
        wp.launch_tiled(
            kernel,
            dim=[worlds, 2],
            inputs=[
                wp.array(adr, device=device),
                wp.array(M, device=device),
                wp.array(elemid, device=device),
                wp.array([0, n], dtype=int, device=device),
                wp.array(y, device=device) if not isinstance(y, wp.array) else y,
                x,
                L_out,
            ],
            device=device,
            block_dim=block_dim,
        )
        meta = kernel.module.load(wp.get_device(device), block_dim).meta
        smem = max(v for key, v in meta.items() if key.endswith("forward_smem_bytes"))
        return L_out.numpy(), x.numpy(), smem >= n * n * 4

    def test_fused_cholesky_gathered(self):
        """MuJoCo Warp's block factorization (gather, factor, store, solve) runs from registers on Metal.

        The stored factor is bit-identical to the same code compiled unfused on Metal, including absent and
        out-of-range gather indices, and the solution matches NumPy.
        """
        rng = np.random.default_rng(9)
        for block_dim in (16, 32):
            for n in sorted({block_dim - 1, block_dim, block_dim + 1, 39, 40, 41}):
                msg = f"n={n}, block_dim={block_dim}"
                problem = _gathered_problem(n, 3, rng)
                factor, x, shared = self._gathered_run(_gathered, n, self.device, block_dim, problem)
                factor_ref, _, shared_ref = self._gathered_run(_gathered_nested, n, self.device, block_dim, problem)
                self.assertEqual(shared, n > 40, msg)
                self.assertTrue(shared_ref, msg)
                np.testing.assert_array_equal(factor, factor_ref, err_msg=msg)
                a, _, _, y, _ = problem
                x_ref = np.linalg.solve(a.astype(np.float64), y[:, :n].astype(np.float64)[..., None])[..., 0]
                np.testing.assert_allclose(x[:, :n], x_ref, rtol=1e-3, atol=1e-4, err_msg=msg)

    def test_fused_cholesky_gathered_leaves_other_shapes_alone(self):
        """Shapes the gathered rewrite must not take keep their shared matrix and compute what the CPU computes."""
        rng = np.random.default_rng(10)
        n = 35
        for make in (
            _gathered_two_loads,
            _gathered_load_of_stored_array,
            _gathered_call_in_offset,
            _gathered_factor_reused,
        ):
            problem = _gathered_problem(n, 2, rng)
            factor, x, shared = self._gathered_run(make, n, self.device, 32, problem)
            factor_cpu, x_cpu, _ = self._gathered_run(make, n, "cpu", 1, problem)
            self.assertTrue(shared, make.__name__)
            np.testing.assert_allclose(factor, factor_cpu, rtol=1e-4, atol=1e-5, err_msg=make.__name__)
            np.testing.assert_allclose(x, x_cpu, rtol=1e-3, atol=1e-4, err_msg=make.__name__)

    def test_fused_cholesky_gathered_store_guard(self):
        """The rewrite moves the factor store past the right-hand side's load, so a load of the same buffer under
        another name raises a device error instead of reading different data."""
        rng = np.random.default_rng(11)
        n = 35
        problem = _gathered_problem(n, 2, rng)
        a, M, elemid, y, adr = problem
        buffer = wp.zeros((2, 2 * n * n + 9), dtype=float, device=self.device)
        buffer.numpy()[:, : 2 * n] = y  # the right-hand sides live in the buffer the factor is stored to
        with self.assertRaisesRegex(RuntimeError, "fused Cholesky: the factor store overlaps"):
            self._gathered_run(_gathered, n, self.device, 32, (a, M, elemid, buffer, adr), L_out=buffer)
        wp.synchronize_device(self.device)  # the error was raised once

    def test_no_64_bit_division_by_runtime_values(self):
        """Compiled Metal kernels divide 64-bit integers only by constants.

        Apple GPUs have no 64-bit integer division, so a division by a runtime value runs a software routine in
        every thread; one such division in the launch index unravel made MuJoCo Warp's grids several times
        slower. This compiles representative kernels with the Metal compiler and checks the IR.
        """
        import re  # noqa: PLC0415
        import subprocess  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        if not _xcrun_metal_available():
            self.skipTest("the Metal compiler (xcrun metal, from Xcode) is not installed")
        device = self.device
        n = 35
        rng = np.random.default_rng(12)
        out2 = wp.zeros((4, 2), dtype=int, device=device)
        cases = []

        wp.launch(launch_coords_2d, dim=(2, 2), inputs=[2, out2], device=device)
        cases.append((launch_coords_2d, 256))
        out3 = wp.zeros((8, 3), dtype=int, device=device)
        wp.launch(launch_coords_3d, dim=(2, 2, 2), inputs=[2, 2, out3], device=device)
        cases.append((launch_coords_3d, 256))
        out4 = wp.zeros((16, 4), dtype=int, device=device)
        wp.launch(launch_coords_4d, dim=(2, 2, 2, 2), inputs=[2, 2, 2, out4], device=device)
        cases.append((launch_coords_4d, 256))
        tiled = wp.zeros((64, 3), dtype=int, device=device)
        wp.launch_tiled(tiled_launch_coords, dim=[2], inputs=[32, tiled], device=device, block_dim=32)
        cases.append((tiled_launch_coords, 32))

        a = wp.array(rng.standard_normal((6, 8)).astype(np.float32), device=device)
        wp.launch(
            strided_view_sum, dim=(6, 4), inputs=[a[:, ::2], wp.zeros(6, dtype=float, device=device)], device=device
        )
        cases.append((strided_view_sum, 256))
        rows = wp.array([0, 2, 4], dtype=int, device=device)
        indexed = wp.indexedarray2d(a, [rows, None])
        wp.launch(
            indexed_copy, dim=(3, 8), inputs=[indexed, wp.zeros((3, 8), dtype=float, device=device)], device=device
        )
        cases.append((indexed_copy, 256))
        wp.launch(
            world_body_dof_grid,
            dim=(16, 30, 35),
            inputs=[
                wp.ones((1, 30), dtype=float, device=device),
                wp.zeros((1, 30), dtype=int, device=device),
                wp.zeros((16, 35), dtype=float, device=device),
            ],
            device=device,
        )
        cases.append((world_body_dof_grid, 256))

        solver = _solver_cholesky(n)
        wp.launch_tiled(
            solver,
            dim=[2],
            inputs=[
                wp.zeros((2, n), dtype=float, device=device),
                wp.array(np.tile(np.eye(n, dtype=np.float32), (2, 1, 1)), device=device),
                wp.zeros(2, dtype=bool, device=device),
                wp.zeros((2, n), dtype=float, device=device),
                wp.zeros(2, dtype=float, device=device),
            ],
            device=device,
            block_dim=32,
        )
        cases.append((solver, 32))
        problem = _gathered_problem(n, 2, rng)
        self._gathered_run(_gathered, n, device, 32, problem)
        cases.append((_gathered(n), 32))

        # divisions whose divisor is not a literal; constant divisors compile to a multiply and shift
        division = re.compile(r"= (?:udiv|urem|sdiv|srem) (?:exact )?i64 [^,]+, (%[\w.]+)")
        with tempfile.TemporaryDirectory() as tmp:
            for kernel, block_dim in cases:
                source = os.path.join(tmp, "kernel.metal")
                with open(source, "w") as f:
                    f.write(_metal_module_source(kernel, block_dim))
                ir = os.path.join(tmp, "kernel.ll")
                subprocess.run(
                    ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-O2", "-fno-fast-math", "-S", "-emit-llvm",
                     source, "-o", ir],
                    check=True,
                    capture_output=True,
                )  # fmt: skip
                with open(ir) as f:
                    found = [line.strip() for line in f if division.search(line)]
                self.assertEqual(found, [], f"{kernel.key}: 64-bit division by a runtime value")

    def test_fused_cholesky_kernels_request_no_matrix_tile(self):
        """MuJoCo Warp's two Cholesky kernel shapes compile without an n x n threadgroup tile on Metal."""
        n = 35
        solver = _solver_cholesky(n)
        wp.launch_tiled(
            solver,
            dim=[1],
            inputs=[
                wp.zeros((1, n), dtype=float, device=self.device),
                wp.array(np.eye(n, dtype=np.float32)[None], device=self.device),
                wp.zeros(1, dtype=bool, device=self.device),
                wp.zeros((1, n), dtype=float, device=self.device),
                wp.zeros(1, dtype=float, device=self.device),
            ],
            device=self.device,
            block_dim=32,
        )
        meta = solver.module.load(wp.get_device(self.device), 32).meta
        solver_bytes = max(v for key, v in meta.items() if key.endswith("forward_smem_bytes"))
        self.assertLess(solver_bytes, n * n * 4)  # the reduction's scratch remains

        _, _, gathered_shared = self._gathered_run(
            _gathered, n, self.device, 32, _gathered_problem(n, 1, np.random.default_rng(13))
        )
        self.assertFalse(gathered_shared)
        meta = _gathered(n).module.load(wp.get_device(self.device), 32).meta
        self.assertEqual(max(v for key, v in meta.items() if key.endswith("forward_smem_bytes")), 0)

    def test_scalar_arguments_are_not_imported(self):
        """NumPy scalars passed by value are not treated as host arrays."""
        device = wp.get_device(self.device)
        a = wp.ones(8, dtype=float, device=self.device)
        wp.launch(scale_kernel, dim=8, inputs=[a, np.float32(3.0)], device=self.device)
        self.assertFalse(device.__dict__.get("_metal_host_memory_pending", False))
        np.testing.assert_array_equal(a.numpy(), np.full(8, 3.0, dtype=np.float32))

    def test_graph_keeps_capture_allocations(self):
        """Arrays allocated during a capture stay valid for replays after Python released them."""
        n = 1024
        x = wp.array(np.arange(n, dtype=np.float32), device=self.device)
        out = wp.zeros(n, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            y = wp.ones(n, dtype=float, device=self.device)  # lives only inside the capture
            wp.launch(saxpy_kernel, dim=n, inputs=[x, y, out], device=self.device)
        del y
        # reuse whatever memory a released buffer would hand back
        junk = [wp.full(n, 7.0, dtype=float, device=self.device) for _ in range(8)]
        out.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(out.numpy(), 2.0 * np.arange(n, dtype=np.float32) + 1.0)
        del junk

    def test_capture_survives_failing_branch(self):
        """A branch body that raises leaves the device usable for normal launches and new captures."""
        cond = wp.ones(1, dtype=int, device=self.device)
        a = wp.zeros(4, dtype=float, device=self.device)

        def failing_body():
            raise ValueError("branch body failed")

        with self.assertRaises(ValueError):
            with wp.ScopedCapture(device=self.device):
                wp.capture_if(cond, on_true=failing_body)
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        wp.capture_launch(capture.graph)
        # one eager launch plus one replay: launches inside a capture are recorded, not executed
        np.testing.assert_array_equal(a.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_replay_translates_nested_arrays_after_new_allocations(self):
        """Graph replays resolve array descriptors stored in device memory after the allocation set changed."""
        values = [wp.array([1.0 + i, 10.0 * (i + 1)], dtype=float, device=self.device) for i in range(4)]
        items = []
        for i, v in enumerate(values):
            h = ArrayHolder()
            h.values = v
            h.offset = float(i)
            items.append(h)
        holders = wp.array(items, dtype=ArrayHolder, device=self.device)
        out = wp.zeros(4, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(holder_sum_kernel, dim=4, inputs=[holders, out], device=self.device)
        extra = [wp.zeros(4096, dtype=float, device=self.device) for _ in range(4)]  # changes the address table
        out.zero_()
        wp.capture_launch(capture.graph)
        expected = np.array([1.0 + i + 10.0 * (i + 1) + i for i in range(4)], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)
        del extra

    def test_print_strings_are_per_kernel(self):
        """String constants with the same generated name in different kernels print their own text."""
        capture = StdOutCapture()
        capture.begin()
        wp.launch(print_first_kernel, dim=1, inputs=[], device=self.device)
        wp.launch(print_second_kernel, dim=1, inputs=[], device=self.device)
        wp.synchronize_device(self.device)
        output = capture.end()
        self.assertIn("metal first string", output)
        self.assertIn("metal second string", output)

    def test_to_torch_attaches_gradient(self):
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("Requires PyTorch")
        a = wp.ones(4, dtype=float, device=self.device, requires_grad=True)
        a.grad.fill_(2.0)
        t = wp.to_torch(a)
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.data_ptr(), a.grad.ptr)
        np.testing.assert_array_equal(t.grad.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_mat44_product_gradients_match_cpu(self):
        """Several 4x4 products in one kernel used to lose the gradient of one operand on Metal."""
        rng = np.random.default_rng(7)
        a_np = (rng.standard_normal((1, 4, 4)) + 3.0 * np.eye(4)).astype(np.float32)
        b_np = rng.standard_normal((1, 4, 4)).astype(np.float32)
        for kernel in (mat44_products_twice, mat44_products_chain, mat44_products_inverse):
            for selected in (0, 17, 31):
                gradients = {}
                for device in ("cpu", self.device):
                    a = wp.array(a_np, dtype=wp.mat44, requires_grad=True, device=device)
                    b = wp.array(b_np, dtype=wp.mat44, requires_grad=True, device=device)
                    out = wp.zeros(32, dtype=float, requires_grad=True, device=device)
                    tape = wp.Tape()
                    with tape:
                        wp.launch(kernel, dim=1, inputs=[a, b], outputs=[out], device=device)
                    seed = np.zeros(32, dtype=np.float32)
                    seed[selected] = 1.0
                    tape.backward(grads={out: wp.array(seed, dtype=float, device=device)})
                    gradients[device] = (tape.gradients[a].numpy(), tape.gradients[b].numpy())
                for on_cpu, on_metal in zip(gradients["cpu"], gradients[self.device], strict=True):
                    np.testing.assert_allclose(
                        on_metal, on_cpu, rtol=1e-3, atol=1e-4, err_msg=f"{kernel.key}[{selected}]"
                    )

    def test_tile_cholesky_larger_than_block(self):
        """A matrix larger than the launch block size gives each lane several columns of the factor.

        The register Cholesky used to zero the mirrored cell, which belongs to another lane's column, so
        the factor was wrong whenever ``n > block_dim`` (up to the size where the register path ends).
        MuJoCo Warp hits this with any model of more than 32 degrees of freedom.
        """
        rng = np.random.default_rng(0)
        worlds = 4
        for n in (27, 33, 35, 40):
            m = rng.standard_normal((worlds, n, n)).astype(np.float32)
            a_np = m @ m.transpose(0, 2, 1) + n * np.eye(n, dtype=np.float32)
            b_np = rng.standard_normal((worlds, n)).astype(np.float32)
            x_ref = np.linalg.solve(a_np.astype(np.float64), b_np.astype(np.float64)[..., None])[..., 0]
            u_ref = np.linalg.cholesky(a_np.astype(np.float64)).transpose(0, 2, 1)
            kernel = _make_cholesky_kernel(n)
            for block_dim in (16, 32, 64):
                a = wp.array(a_np, dtype=float, device=self.device)
                b = wp.array(b_np, dtype=float, device=self.device)
                factor = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                x = wp.zeros((worlds, n), dtype=float, device=self.device)
                wp.launch_tiled(kernel, dim=[worlds], inputs=[a, b, factor, x], device=self.device, block_dim=block_dim)
                msg = f"n={n}, block_dim={block_dim}"
                np.testing.assert_allclose(factor.numpy(), u_ref, rtol=1e-4, atol=1e-4, err_msg=msg)
                np.testing.assert_allclose(x.numpy(), x_ref, rtol=1e-3, atol=1e-4, err_msg=msg)

    def test_tile_ops_across_block_boundary(self):
        """Tile sizes just below, at and above the block size, and above two blocks.

        Operations with a Metal-specific implementation split a tile over the lanes of a block, so the
        interesting sizes are the ones where the number of elements per lane changes.
        """
        rng = np.random.default_rng(2)
        worlds = 3
        for block_dim in (16, 32, 64):
            for n in (block_dim - 1, block_dim, block_dim + 1, 2 * block_dim + 1):
                a_np = rng.standard_normal((worlds, n)).astype(np.float32)
                keys_np = rng.permutation(worlds * n).reshape(worlds, n).astype(np.int32)
                expected = {
                    "total": a_np.sum(1, keepdims=True),
                    "lowest": a_np.min(1, keepdims=True),
                    "highest": a_np.max(1, keepdims=True),
                    "arg_lowest": a_np.argmin(1)[:, None],
                    "arg_highest": a_np.argmax(1)[:, None],
                    "running": np.cumsum(a_np.astype(np.float64), 1),
                    "keys_out": np.sort(keys_np, 1),
                    "order": np.argsort(keys_np, 1),
                }
                out = {
                    name: wp.zeros(ref.shape, dtype=int if ref.dtype.kind == "i" else float, device=self.device)
                    for name, ref in expected.items()
                }
                wp.launch_tiled(
                    _make_tile_ops_kernel(n),
                    dim=[worlds],
                    inputs=[
                        wp.array(a_np, dtype=float, device=self.device),
                        wp.array(keys_np, dtype=int, device=self.device),
                        *out.values(),
                    ],
                    device=self.device,
                    block_dim=block_dim,
                )
                for name, ref in expected.items():
                    np.testing.assert_allclose(
                        out[name].numpy(), ref, rtol=1e-4, atol=1e-4, err_msg=f"{name}, n={n}, block_dim={block_dim}"
                    )

    def test_tile_matmul_across_block_boundary(self):
        rng = np.random.default_rng(3)
        worlds = 3
        for block_dim in (16, 32):
            for n in (block_dim - 1, block_dim, block_dim + 1):
                a_np = rng.standard_normal((worlds, n, n)).astype(np.float32)
                b_np = rng.standard_normal((worlds, n, n)).astype(np.float32)
                product = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                acc = wp.zeros((worlds, n, n), dtype=float, device=self.device)
                wp.launch_tiled(
                    _make_tile_matmul_kernel(n),
                    dim=[worlds],
                    inputs=[
                        wp.array(a_np, dtype=float, device=self.device),
                        wp.array(b_np, dtype=float, device=self.device),
                        product,
                        acc,
                    ],
                    device=self.device,
                    block_dim=block_dim,
                )
                ref = a_np.astype(np.float64) @ b_np
                msg = f"n={n}, block_dim={block_dim}"
                np.testing.assert_allclose(product.numpy(), ref, rtol=1e-3, atol=1e-3, err_msg=msg)
                np.testing.assert_allclose(acc.numpy(), a_np + ref, rtol=1e-3, atol=1e-3, err_msg=msg)


class TestMetalInlineBudget(unittest.TestCase):
    """Forced inlining is bounded by the fully inlined size (pure codegen logic, no device needed)."""

    def test_budget_counts_callees(self):
        from warp._src import codegen  # noqa: PLC0415

        budget = codegen._METAL_FORCE_INLINE_MAX_LINES
        names = ("wp_test_inline_small", "wp_test_inline_big", "wp_test_inline_caller")
        try:
            self.assertTrue(codegen._metal_force_inline(names[0], "x = 1;\n" * 10))
            self.assertFalse(codegen._metal_force_inline(names[1], "x = 1;\n" * (budget + 1)))
            # two lines of its own, but inlining its callee would exceed the budget
            self.assertFalse(codegen._metal_force_inline(names[2], f"y = {names[1]}(a);\nreturn y;\n"))
            self.assertTrue(codegen._metal_force_inline(names[2], f"y = {names[0]}(a);\nreturn y;\n"))
        finally:
            for name in names:
                codegen._metal_inlined_lines.pop(name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
