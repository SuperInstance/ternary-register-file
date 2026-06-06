# ternary-register-file

[![crate](https://img.shields.io/badge/crate-ternary--register--file-blue)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)](./LICENSE)

Register file allocation for **ternary GPU kernels** — a CPU-side compiler simulation that models physical register management for ternary (base-3) packed values.

## Overview

In ternary computing, values are represented as sequences of **trits** (ternary digits: −1, 0, +1) rather than binary bits. This denser encoding means fewer physical registers are needed per value, but register allocation must account for variable-width packed trit groups.

This crate provides a complete register file simulation for ternary GPU kernel compilation:

- **`RegisterFile`** — tracks physical register usage per thread, with allocate/free, utilization metrics, and contiguous allocation support.
- **`RegisterAllocator`** — linear-scan register allocator that processes virtual registers in live-range order, automatically expiring dead intervals and spilling when pressure exceeds capacity.
- **`LiveRangeAnalysis`** — computes live ranges from def-use chains, builds interference graphs, and measures pressure at arbitrary program points.
- **`RegisterPressureEstimator`** — computes max/average pressure across the program and identifies pressure hotspots where spills are likely.
- **`SpillCostCalculator`** — estimates spill cost using live-range length × register count, with weighted variants that factor in loop depth and reuse frequency.

## Why Ternary?

Ternary arithmetic is more compact: 20 trits fit in a single 32-bit register (vs. 32 bits for binary). Packed ternary values reduce register pressure significantly, enabling more values live simultaneously. This crate quantifies that advantage with concrete register counts and spill cost analysis.

## Quick Start

```rust
use ternary_register_file::*;

// Create an allocator with 64 physical registers
let mut allocator = RegisterAllocator::new(64);

// Request registers for ternary values
let v0 = allocator.request(LiveRange::new(0, 20), 20, false); // 1 register
let v1 = allocator.request(LiveRange::new(5, 30), 40, false); // 2 registers
let v2 = allocator.request(LiveRange::new(15, 25), 10, true);  // 1 register (packed)

// Run allocation
let result = allocator.allocate_all();
println!("Allocated: {}", result.allocated_count());
println!("Spilled:   {}", result.spilled_count());
println!("Peak:      {} registers", result.peak_utilization);
```

## Live Range Analysis

```rust
use ternary_register_file::*;

let ranges = vec![
    (VirtualReg(0), LiveRange::new(0, 10)),
    (VirtualReg(1), LiveRange::new(5, 15)),
    (VirtualReg(2), LiveRange::new(20, 30)),
];

// Build interference graph
let graph = LiveRangeAnalysis::build_interference_graph(&ranges);
assert!(graph[&VirtualReg(0)].contains(&VirtualReg(1))); // they overlap
assert!(!graph[&VirtualReg(0)].contains(&VirtualReg(2))); // no overlap

// Pressure at a point
let pressure_ranges = vec![
    (VirtualReg(0), LiveRange::new(0, 10), 2u32),
    (VirtualReg(1), LiveRange::new(5, 15), 3u32),
];
assert_eq!(LiveRangeAnalysis::pressure_at_point(&pressure_ranges, 7), 5);
```

## Spill Cost Analysis

```rust
use ternary_register_file::*;

let lr = LiveRange::new(0, 100);
let cost = SpillCostCalculator::spill_cost(&lr, 4);
// cost = 100 × 4 = 400.0

let weighted = SpillCostCalculator::weighted_spill_cost(&lr, 4, 5, 2);
// Includes loop depth factor (100×) and reuse factor (6×)
```

## Key Types

| Type | Description |
|------|-------------|
| `RegisterFile` | Physical register pool with allocation tracking |
| `RegisterAllocator` | Linear-scan allocator with automatic live-range expiry |
| `LiveRange` | Instruction interval `[start, end)` |
| `AllocationRequest` | Virtual register request with trit count |
| `AllocationResult` | Outcome: successful allocations + spilled requests |
| `Allocation` | Physical register assignment for a virtual register |

## Constants

- `TRITS_PER_REGISTER = 20` — Number of ternary trits that fit in a 32-bit register (⌈log₃(2³²)⌉).

## Testing

```bash
cargo test
```

17 comprehensive tests covering allocation, live ranges, pressure estimation, spill costs, packed vs. unpacked trit efficiency, interference graphs, and utilization tracking.

## License

MIT OR Apache-2.0
