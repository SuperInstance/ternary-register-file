# ternary-register-file

Register file allocation for ternary GPU kernels.

GPUs have a fixed number of registers per streaming multiprocessor (SM). When a kernel needs more registers than available, the compiler spills to local memory (which is really L1/L2 cache, much slower). This crate simulates register allocation for ternary kernels, where packed trit values are denser than binary — 20 trits fit in a single 32-bit register — so you can predict register pressure, plan spills, and optimize your kernel before running it on hardware.

The key insight: ternary values are more compact than binary. A 32-bit register holds `⌈log₃(2³²)⌉ = 20` trits (vs. 32 bits). This means a ternary kernel working with the same number of *values* uses fewer registers than a binary kernel. Fewer registers → higher occupancy → better throughput.

## Why This Exists

Register allocation is the most important optimization in GPU kernel performance. Too many registers per thread → fewer threads per SM → lower occupancy → memory latency isn't hidden. The compiler makes allocation decisions, but it doesn't know about your high-level intent — which values are live simultaneously, which loops contain the critical path, which variables you'd prefer to spill.

This crate gives you:

1. **A register file model** — `RegisterFile` tracks physical register usage with allocate/free
2. **A linear-scan allocator** — `RegisterAllocator` processes virtual registers in live-range order
3. **Live range analysis** — build interference graphs, compute pressure at any program point
4. **Pressure estimation** — find hotspots where spills are likely
5. **Spill cost analysis** — decide *what* to spill when you must

## Quick Start

```rust
use ternary_register_file::*;

// Model a GPU with 64 registers per thread
let mut allocator = RegisterAllocator::new(64);

// Request registers for variables in your kernel
let v0 = allocator.request(LiveRange::new(0, 20), 20, false);  // 1 register (20 trits)
let v1 = allocator.request(LiveRange::new(5, 30), 40, false);  // 2 registers (40 trits)
let v2 = allocator.request(LiveRange::new(15, 25), 10, true);  // 1 register (packed)
let v3 = allocator.request(LiveRange::new(0, 30), 60, false);  // 3 registers

// Run allocation
let result = allocator.allocate_all();

println!("Allocated: {}", result.allocated_count());  // 4
println!("Spilled:   {}", result.spilled_count());    // 0 (fits in 64 regs)
println!("Peak usage: {} registers", result.peak_utilization);
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 RegisterAllocator                         │
│                                                          │
│  request(live_range, trits, packed) → VirtualReg         │
│  allocate_all() → AllocationResult                       │
│                                                          │
│  ┌──────────────┐    ┌──────────────────────┐           │
│  │ RegisterFile │    │ Live Range Tracking  │           │
│  │ 64 physical  │    │ sorted by start      │           │
│  │ registers    │    │ active intervals     │           │
│  └──────────────┘    └──────────────────────┘           │
├──────────────────────────────────────────────────────────┤
│  LiveRangeAnalysis                                       │
│  compute_from_def_use(def, uses) → LiveRange             │
│  build_interference_graph(ranges) → adjacency list       │
│  pressure_at_point(ranges, point) → register count       │
├──────────────────────────────────────────────────────────┤
│  RegisterPressureEstimator                               │
│  max_pressure(ranges) → peak register count              │
│  avg_pressure(ranges) → average across program           │
│  pressure_hotspots(ranges, threshold) → problem points   │
├──────────────────────────────────────────────────────────┤
│  SpillCostCalculator                                     │
│  spill_cost(live_range, regs) → f64                      │
│  weighted_spill_cost(range, regs, reuse, loop_depth) → f64│
│  select_spill_candidate(candidates) → cheapest to spill  │
└──────────────────────────────────────────────────────────┘
```

### RegisterFile

The physical register pool. Tracks which registers are free, which are allocated, and to which virtual register:

```rust
let mut rf = RegisterFile::new(32);

// Allocate 4 registers (not necessarily contiguous)
let regs = rf.allocate(4).unwrap();
assert_eq!(rf.free_count(), 28);
assert_eq!(rf.used_count(), 4);

// Allocate contiguous registers (for vector operations)
let contig = rf.allocate_contiguous(4).unwrap();
// regs are [r0, r1, r2, r3] — guaranteed adjacent

// Free them
rf.free(&regs);
assert_eq!(rf.free_count(), 32);

// Utilization
rf.utilization(); // 0.0 = empty, 1.0 = full
```

### Linear-Scan Allocation

The allocator sorts virtual registers by live range start, then processes them in order:

1. **Expire** any intervals that ended before the current one starts
2. **Try to allocate** physical registers for the new interval
3. **Spill** if no registers are available

```rust
let mut allocator = RegisterAllocator::new(64);

// These three overlap: v0 and v1 are live at instruction 5
allocator.request(LiveRange::new(0, 10), 20, false);   // 1 reg
allocator.request(LiveRange::new(5, 15), 20, false);   // 1 reg  
allocator.request(LiveRange::new(12, 20), 20, false);  // 1 reg
// Peak pressure: 2 registers (v0+v1 overlap, then v1+v2)

let result = allocator.allocate_all();
assert_eq!(result.peak_utilization, 2);
```

### Live Range Analysis

Build an interference graph to see which variables can't share a register:

```rust
let ranges = vec![
    (VirtualReg(0), LiveRange::new(0, 10)),
    (VirtualReg(1), LiveRange::new(5, 15)),
    (VirtualReg(2), LiveRange::new(20, 30)),
];

let graph = LiveRangeAnalysis::build_interference_graph(&ranges);
// v0 and v1 interfere (overlap at instructions 5-9)
// v0 and v2 don't interfere (no overlap)
// → v0 and v2 could share a register
```

### Pressure Estimation

Find where register demand peaks:

```rust
let ranges = vec![
    (LiveRange::new(0, 10), 2),   // uses 2 registers
    (LiveRange::new(5, 15), 3),   // uses 3 registers
    (LiveRange::new(0, 5), 1),    // uses 1 register
];

// Max pressure: at instruction 5-9, total = 2+3 = 5 registers
assert_eq!(RegisterPressureEstimator::max_pressure(&ranges), 5);

// Find hotspots exceeding threshold
let hotspots = RegisterPressureEstimator::pressure_hotspots(&ranges, 4);
// Instructions 5-9 exceed 4 registers
```

### Spill Cost Analysis

When you can't fit everything in registers, spill the cheapest variable:

```rust
// Basic cost: live_range_length × register_count
let cost = SpillCostCalculator::spill_cost(&LiveRange::new(0, 100), 4);
// cost = 100 × 4.0 = 400.0

// Weighted: accounts for loop nesting and reuse
let weighted = SpillCostCalculator::weighted_spill_cost(
    &LiveRange::new(0, 10),  // 10 instructions
    2,                        // 2 registers
    5,                        // used 5 times
    2,                        // nested 2 loops deep
);
// weighted = 10 × 2 × 10² × 6 = 12,000

// Select cheapest spill candidate
let candidates = vec![
    (VirtualReg(0), LiveRange::new(0, 2), 1, 10.0),   // short-lived
    (VirtualReg(1), LiveRange::new(0, 100), 4, 400.0), // long-lived
];
let best = SpillCostCalculator::select_spill_candidate(&candidates);
assert_eq!(best, Some(VirtualReg(0))); // spill the cheap one
```

## Trits Per Register

The fundamental constant: `TRITS_PER_REGISTER = 20`.

Why 20? A 32-bit register can represent values 0 to 2³²−1. In balanced ternary, the largest value representable with n trits is `(3ⁿ − 1) / 2`. We need `3ⁿ ≥ 2³²`, so `n ≥ log₃(2³²) ≈ 20.19`. Thus 20 trits fit in one register (the 21st trit would require 33 bits).

Practical impact: A value that needs 32 bits in binary (one register) can hold 20 ternary digits. If your ternary network uses 10-trit weights, you can fit 2 weights per register — cutting register pressure in half compared to naive binary storage.

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `RegisterFile` | Physical register pool |
| `RegisterAllocator` | Linear-scan allocator |
| `LiveRange` | Instruction interval `[start, end)` |
| `VirtualReg` | Compiler-assigned virtual register ID |
| `RegisterId` | Physical register ID |
| `AllocationRequest` | Virtual register with trit count and live range |
| `AllocationResult` | Allocation outcome: successes + spills |

### `AllocationResult`

| Field | Description |
|-------|-------------|
| `allocations` | Successfully allocated virtual registers |
| `spilled` | Virtual registers that didn't fit |
| `total_registers` | Physical register count |
| `peak_utilization` | Max registers used at any point |

## Real-World Example: Analyzing a Ternary GEMM Kernel

```rust
use ternary_register_file::*;

// A ternary matrix multiply kernel
// Each thread computes one output element: C[i][j] = sum(A[i][k] * B[k][j])
// A and B rows/columns are 64 trits each

let mut allocator = RegisterAllocator::new(255); // typical GPU limit

// Accumulator (float, 1 register)
let acc = allocator.request(LiveRange::new(0, 100), 20, false);

// A row tile (64 trits = 4 registers, live for entire inner loop)
let a_row = allocator.request(LiveRange::new(0, 100), 64, true);

// B column tile (64 trits = 4 registers, live for entire inner loop)
let b_col = allocator.request(LiveRange::new(0, 100), 64, true);

// Loop counter and index variables (1 register each)
let idx = allocator.request(LiveRange::new(0, 100), 20, false);
let tmp = allocator.request(LiveRange::new(10, 90), 20, false);

let result = allocator.allocate_all();

println!("Allocated: {} virtual registers", result.allocated_count());
println!("Peak physical registers: {}", result.peak_utilization);
println!("Spilled: {}", result.spilled_count());

if result.spilled_count() > 0 {
    let spill_cost = SpillCostCalculator::total_spill_cost(&result.spilled);
    println!("Total spill cost: {:.0}", spill_cost);
}
```

## Ecosystem Connections

- **`ternary-warp-block`** — Warp operations produce values that need register allocation
- **`ternary-shared-memory`** — Values that don't fit in registers spill to shared memory
- **`ternary-grid-launch`** — Register count per thread affects occupancy calculations
- **`ternary-constant-cache`** — Read-only data that doesn't need registers (cached separately)

## Performance Notes

- **Allocation**: O(n log n) for sorting requests + O(n) for the scan. Fast for kernel-sized inputs (<1000 virtual registers).
- **Interference graph**: O(n²) for the pairwise overlap check. Fine for n < 1000.
- **Pressure estimation**: O(max_instruction × n). Could be O(n log n) with a sweep-line algorithm.
- **The 20-trit advantage**: Compared to binary, ternary uses ~37% fewer registers for the same number of values. This translates directly to higher occupancy.

## Open Questions

- **Graph coloring allocator**: Linear scan is simple but suboptimal. A graph-coloring allocator (like LLVM's) would produce better allocations at the cost of complexity.
- **Coalescing**: No register coalescing for copy instructions. Would reduce register pressure for values that are just moved around.
- **Split intervals**: No support for splitting a live range into parts (partially in registers, partially spilled). This is how real compilers handle high pressure.

## License

MIT OR Apache-2.0
