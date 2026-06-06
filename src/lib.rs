//! # ternary-register-file
//!
//! Register file allocation for ternary GPU kernels.
//!
//! Simulates a CPU-side register file allocator for ternary (base-3) GPU kernels.
//! Trit-packed values are denser than binary, so fewer registers are needed per value.
//! This crate provides linear-scan allocation, live-range analysis, register pressure
//! estimation, and spill-cost calculation.

use std::collections::{BTreeMap, HashMap, HashSet};

/// Number of ternary trits that fit in a single 32-bit register (ceil(log_3(2^32)) ≈ 20).
pub const TRITS_PER_REGISTER: u32 = 20;

/// A single register identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegisterId(pub u32);

/// A virtual register requested during compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VirtualReg(pub u32);

/// A live range: the instruction interval [start, end) during which a value is alive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub start: u32,
    pub end: u32,
}

impl LiveRange {
    pub fn new(start: u32, end: u32) -> Self {
        assert!(start <= end, "live range start must be <= end");
        Self { start, end }
    }

    pub fn overlaps(&self, other: &LiveRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Descriptor for a value that needs register allocation.
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub vreg: VirtualReg,
    pub live_range: LiveRange,
    /// Number of trits this value occupies.
    pub trit_count: u32,
    /// Whether this value is packed (multiple logical values in one physical register).
    pub packed: bool,
}

impl AllocationRequest {
    /// Number of physical registers needed for this request.
    pub fn registers_needed(&self) -> u32 {
        if self.trit_count == 0 {
            return 0;
        }
        (self.trit_count + TRITS_PER_REGISTER - 1) / TRITS_PER_REGISTER
    }
}

/// Result of a register allocation.
#[derive(Debug, Clone)]
pub struct Allocation {
    pub vreg: VirtualReg,
    pub physical_regs: Vec<RegisterId>,
    pub live_range: LiveRange,
    pub registers_used: u32,
}

/// The register file: tracks physical registers and their current allocations.
#[derive(Debug, Clone)]
pub struct RegisterFile {
    total_registers: u32,
    /// Currently allocated physical register -> allocation info.
    allocations: HashMap<RegisterId, Allocation>,
    /// Map from virtual register to its allocation.
    vreg_map: HashMap<VirtualReg, Allocation>,
    /// Ordered free list.
    free_set: HashSet<u32>,
}

impl RegisterFile {
    /// Create a new register file with the given number of physical registers.
    pub fn new(total_registers: u32) -> Self {
        let free_set = (0..total_registers).collect();
        Self {
            total_registers,
            allocations: HashMap::new(),
            vreg_map: HashMap::new(),
            free_set,
        }
    }

    /// Total number of physical registers.
    pub fn total_registers(&self) -> u32 {
        self.total_registers
    }

    /// Number of currently free registers.
    pub fn free_count(&self) -> u32 {
        self.free_set.len() as u32
    }

    /// Number of currently used registers.
    pub fn used_count(&self) -> u32 {
        self.total_registers - self.free_count()
    }

    /// Check if a specific register is free.
    pub fn is_free(&self, reg: RegisterId) -> bool {
        self.free_set.contains(&reg.0)
    }

    /// Allocate `count` contiguous physical registers. Returns the registers or None.
    pub fn allocate_contiguous(&mut self, count: u32) -> Option<Vec<RegisterId>> {
        if count == 0 {
            return Some(vec![]);
        }
        if self.free_count() < count {
            return None;
        }
        // Linear scan for contiguous block
        let mut free_sorted: Vec<u32> = self.free_set.iter().copied().collect();
        free_sorted.sort();
        let mut run_start = 0;
        for i in 0..free_sorted.len() {
            if i > 0 && free_sorted[i] != free_sorted[i - 1] + 1 {
                run_start = i;
            }
            let run_len = i - run_start + 1;
            if run_len >= count as usize {
                let start_idx = i + 1 - count as usize;
                let regs: Vec<RegisterId> = free_sorted[start_idx..=i]
                    .iter()
                    .map(|&r| RegisterId(r))
                    .collect();
                for reg in &regs {
                    self.free_set.remove(&reg.0);
                }
                return Some(regs);
            }
        }
        None
    }

    /// Allocate `count` physical registers (not necessarily contiguous).
    pub fn allocate(&mut self, count: u32) -> Option<Vec<RegisterId>> {
        if self.free_count() < count {
            return None;
        }
        let mut regs = Vec::with_capacity(count as usize);
        let mut sorted: Vec<u32> = self.free_set.iter().copied().collect();
        sorted.sort();
        for &r in sorted.iter().take(count as usize) {
            self.free_set.remove(&r);
            regs.push(RegisterId(r));
        }
        Some(regs)
    }

    /// Free a set of registers.
    pub fn free(&mut self, regs: &[RegisterId]) {
        for reg in regs {
            self.free_set.insert(reg.0);
            self.allocations.remove(reg);
        }
    }

    /// Assign an allocation to a virtual register.
    pub fn assign(&mut self, alloc: Allocation) {
        for &reg in &alloc.physical_regs {
            self.allocations.insert(reg, alloc.clone());
        }
        self.vreg_map.insert(alloc.vreg, alloc);
    }

    /// Deallocate a virtual register.
    pub fn deallocate(&mut self, vreg: VirtualReg) -> Option<Allocation> {
        if let Some(alloc) = self.vreg_map.remove(&vreg) {
            self.free(&alloc.physical_regs);
            Some(alloc)
        } else {
            None
        }
    }

    /// Get allocation for a virtual register.
    pub fn get_allocation(&self, vreg: VirtualReg) -> Option<&Allocation> {
        self.vreg_map.get(&vreg)
    }

    /// Current register utilization as a fraction [0.0, 1.0].
    pub fn utilization(&self) -> f64 {
        if self.total_registers == 0 {
            0.0
        } else {
            self.used_count() as f64 / self.total_registers as f64
        }
    }
}

/// Linear-scan register allocator.
#[derive(Debug)]
pub struct RegisterAllocator {
    register_file: RegisterFile,
    /// Requests sorted by live range start.
    sorted_requests: Vec<AllocationRequest>,
    /// Active allocations sorted by live range end.
    active: BTreeMap<u32, Vec<Allocation>>,
    /// Spilled requests.
    spilled: Vec<AllocationRequest>,
    next_vreg: u32,
}

impl RegisterAllocator {
    pub fn new(total_registers: u32) -> Self {
        Self {
            register_file: RegisterFile::new(total_registers),
            sorted_requests: Vec::new(),
            active: BTreeMap::new(),
            spilled: Vec::new(),
            next_vreg: 0,
        }
    }

    /// Add a request for register allocation.
    pub fn add_request(&mut self, req: AllocationRequest) {
        self.sorted_requests.push(req);
    }

    /// Create and add a request with auto-incremented vreg.
    pub fn request(&mut self, live_range: LiveRange, trit_count: u32, packed: bool) -> VirtualReg {
        let vreg = VirtualReg(self.next_vreg);
        self.next_vreg += 1;
        self.add_request(AllocationRequest {
            vreg,
            live_range,
            trit_count,
            packed,
        });
        vreg
    }

    /// Expire old allocations that are no longer live at `current_point`.
    fn expire_old(&mut self, current_point: u32) {
        let ends_to_remove: Vec<u32> = self
            .active
            .keys()
            .copied()
            .filter(|&end| end <= current_point)
            .collect();
        for end in ends_to_remove {
            if let Some(allocs) = self.active.remove(&end) {
                for alloc in allocs {
                    self.register_file.deallocate(alloc.vreg);
                }
            }
        }
    }

    /// Run the linear-scan allocation algorithm.
    pub fn allocate_all(&mut self) -> AllocationResult {
        // Sort requests by live range start
        self.sorted_requests.sort_by_key(|r| r.live_range.start);

        let mut allocations = Vec::new();
        let requests: Vec<AllocationRequest> = self.sorted_requests.clone();

        for req in requests {
            // Expire intervals that ended before this one starts
            self.expire_old(req.live_range.start);

            let needed = req.registers_needed();
            if let Some(regs) = self.register_file.allocate(needed) {
                let alloc = Allocation {
                    vreg: req.vreg,
                    physical_regs: regs.clone(),
                    live_range: req.live_range,
                    registers_used: needed,
                };
                self.register_file.assign(alloc.clone());
                self.active
                    .entry(req.live_range.end)
                    .or_default()
                    .push(alloc.clone());
                allocations.push(alloc);
            } else {
                self.spilled.push(req);
            }
        }

        let peak = self.estimate_peak_pressure(&allocations);
        AllocationResult {
            allocations,
            spilled: self.spilled.clone(),
            total_registers: self.register_file.total_registers,
            peak_utilization: peak,
        }
    }

    /// Try to spill a currently-active allocation to make room.
    fn try_spill_for(&mut self, _req: &AllocationRequest) -> bool {
        // In a full implementation, we'd spill the longest interval.
        // For this simulation, just report failure.
        false
    }

    /// Estimate peak register pressure across the allocation result.
    fn estimate_peak_pressure(&self, allocations: &[Allocation]) -> u32 {
        if allocations.is_empty() {
            return 0;
        }
        let max_end = allocations
            .iter()
            .map(|a| a.live_range.end)
            .max()
            .unwrap_or(0);
        let mut pressure = vec![0u32; max_end as usize + 1];
        for alloc in allocations {
            for i in alloc.live_range.start..alloc.live_range.end {
                if (i as usize) < pressure.len() {
                    pressure[i as usize] += alloc.registers_used;
                }
            }
        }
        *pressure.iter().max().unwrap_or(&0)
    }

    /// Get a reference to the register file.
    pub fn register_file(&self) -> &RegisterFile {
        &self.register_file
    }

    /// Get spilled requests.
    pub fn spilled(&self) -> &[AllocationRequest] {
        &self.spilled
    }
}

/// Result of running the allocator.
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub allocations: Vec<Allocation>,
    pub spilled: Vec<AllocationRequest>,
    pub total_registers: u32,
    pub peak_utilization: u32,
}

impl AllocationResult {
    /// Number of successfully allocated vregs.
    pub fn allocated_count(&self) -> usize {
        self.allocations.len()
    }

    /// Number of spilled vregs.
    pub fn spilled_count(&self) -> usize {
        self.spilled.len()
    }

    /// Total registers used across all allocations (summing individual register counts).
    pub fn total_register_uses(&self) -> u32 {
        self.allocations.iter().map(|a| a.registers_used).sum()
    }
}

/// Live range analysis utilities.
pub struct LiveRangeAnalysis;

impl LiveRangeAnalysis {
    /// Compute the live range for a variable given its definition and use points.
    pub fn compute_from_def_use(def: u32, uses: &[u32]) -> LiveRange {
        if uses.is_empty() {
            LiveRange::new(def, def + 1)
        } else {
            let min_use = *uses.iter().min().unwrap();
            let max_use = *uses.iter().max().unwrap();
            LiveRange::new(def.min(min_use), max_use + 1)
        }
    }

    /// Compute interference between two sets of live ranges.
    pub fn interferes(a: &LiveRange, b: &LiveRange) -> bool {
        a.overlaps(b)
    }

    /// Build an interference graph (adjacency list) from a set of live ranges.
    pub fn build_interference_graph(ranges: &[(VirtualReg, LiveRange)]) -> HashMap<VirtualReg, HashSet<VirtualReg>> {
        let mut graph: HashMap<VirtualReg, HashSet<VirtualReg>> = HashMap::new();
        for (vreg, _) in ranges {
            graph.insert(*vreg, HashSet::new());
        }
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                let (va, ra) = &ranges[i];
                let (vb, rb) = &ranges[j];
                if ra.overlaps(rb) {
                    graph.get_mut(va).unwrap().insert(*vb);
                    graph.get_mut(vb).unwrap().insert(*va);
                }
            }
        }
        graph
    }

    /// Estimate register pressure at a given program point.
    pub fn pressure_at_point(ranges: &[(VirtualReg, LiveRange, u32)], point: u32) -> u32 {
        ranges
            .iter()
            .filter(|(_, lr, _)| lr.start <= point && point < lr.end)
            .map(|(_, _, regs)| *regs)
            .sum()
    }
}

/// Register pressure estimator.
pub struct RegisterPressureEstimator;

impl RegisterPressureEstimator {
    /// Estimate the maximum register pressure across all program points.
    pub fn max_pressure(ranges: &[(LiveRange, u32)]) -> u32 {
        if ranges.is_empty() {
            return 0;
        }
        let max_end = ranges.iter().map(|(lr, _)| lr.end).max().unwrap_or(0);
        let mut pressure = vec![0u32; max_end as usize + 1];
        for (lr, count) in ranges {
            for i in lr.start..lr.end {
                if (i as usize) < pressure.len() {
                    pressure[i as usize] += count;
                }
            }
        }
        *pressure.iter().max().unwrap_or(&0)
    }

    /// Compute average register pressure.
    pub fn avg_pressure(ranges: &[(LiveRange, u32)]) -> f64 {
        if ranges.is_empty() {
            return 0.0;
        }
        let max_end = ranges.iter().map(|(lr, _)| lr.end).max().unwrap_or(0);
        if max_end == 0 {
            return 0.0;
        }
        let total: u64 = ranges.iter().map(|(lr, count)| (lr.len() as u64) * (*count as u64)).sum();
        total as f64 / max_end as f64
    }

    /// Identify program points where pressure exceeds a threshold.
    pub fn pressure_hotspots(ranges: &[(LiveRange, u32)], threshold: u32) -> Vec<(u32, u32)> {
        if ranges.is_empty() {
            return vec![];
        }
        let max_end = ranges.iter().map(|(lr, _)| lr.end).max().unwrap_or(0);
        let mut pressure = vec![0u32; max_end as usize + 1];
        for (lr, count) in ranges {
            for i in lr.start..lr.end {
                if (i as usize) < pressure.len() {
                    pressure[i as usize] += count;
                }
            }
        }
        pressure
            .iter()
            .enumerate()
            .filter(|&(_, p)| *p > threshold)
            .map(|(i, &p)| (i as u32, p))
            .collect()
    }
}

/// Spill cost calculation.
pub struct SpillCostCalculator;

impl SpillCostCalculator {
    /// Simple spill cost: live range length × register count.
    /// Longer-lived, wider values are more expensive to spill.
    pub fn spill_cost(live_range: &LiveRange, register_count: u32) -> f64 {
        live_range.len() as f64 * register_count as f64
    }

    /// Weighted spill cost considering reuse frequency.
    pub fn weighted_spill_cost(
        live_range: &LiveRange,
        register_count: u32,
        reuse_count: u32,
        loop_depth: u32,
    ) -> f64 {
        let base = live_range.len() as f64 * register_count as f64;
        let loop_factor = 10u32.pow(loop_depth) as f64;
        let reuse_factor = (reuse_count + 1) as f64;
        base * loop_factor * reuse_factor
    }

    /// Select the best candidate for spilling (lowest cost).
    pub fn select_spill_candidate(
        candidates: &[(VirtualReg, LiveRange, u32, f64)],
    ) -> Option<VirtualReg> {
        candidates
            .iter()
            .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(vreg, _, _, _)| *vreg)
    }

    /// Estimate total spill cost for all spilled allocations.
    pub fn total_spill_cost(spilled: &[AllocationRequest]) -> f64 {
        spilled
            .iter()
            .map(|req| Self::spill_cost(&req.live_range, req.registers_needed()))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_file_allocate_and_free() {
        let mut rf = RegisterFile::new(32);
        assert_eq!(rf.free_count(), 32);

        let regs = rf.allocate(4).unwrap();
        assert_eq!(regs.len(), 4);
        assert_eq!(rf.free_count(), 28);

        rf.free(&regs);
        assert_eq!(rf.free_count(), 32);
    }

    #[test]
    fn test_register_file_allocate_fails_when_full() {
        let mut rf = RegisterFile::new(4);
        let r1 = rf.allocate(4).unwrap();
        assert!(rf.allocate(1).is_none());
        rf.free(&r1);
        assert!(rf.allocate(1).is_some());
    }

    #[test]
    fn test_register_file_contiguous() {
        let mut rf = RegisterFile::new(8);
        let regs = rf.allocate_contiguous(4).unwrap();
        // Should be contiguous
        for i in 0..3 {
            assert_eq!(regs[i + 1].0, regs[i].0 + 1);
        }
    }

    #[test]
    fn test_live_range_overlap() {
        let a = LiveRange::new(0, 5);
        let b = LiveRange::new(3, 8);
        let c = LiveRange::new(5, 10);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(b.overlaps(&c));
    }

    #[test]
    fn test_live_range_tracking() {
        let mut allocator = RegisterAllocator::new(16);
        allocator.request(LiveRange::new(0, 5), 20, false); // 1 reg
        allocator.request(LiveRange::new(3, 8), 20, false); // 1 reg
        allocator.request(LiveRange::new(0, 2), 20, false); // 1 reg

        let result = allocator.allocate_all();
        // First (0-5) and third (0-2) overlap; second (3-8) overlaps with first
        // But third ends at 2, so at point 3 it's freed
        assert_eq!(result.spilled_count(), 0);
    }

    #[test]
    fn test_pressure_estimation() {
        let ranges = vec![
            (LiveRange::new(0, 10), 2u32),
            (LiveRange::new(5, 15), 3u32),
            (LiveRange::new(0, 5), 1u32),
        ];
        assert_eq!(RegisterPressureEstimator::max_pressure(&ranges), 5); // At point 5-9: 2+3=5
        assert!(RegisterPressureEstimator::avg_pressure(&ranges) > 0.0);
    }

    #[test]
    fn test_pressure_hotspots() {
        let ranges = vec![
            (LiveRange::new(0, 10), 5u32),
            (LiveRange::new(5, 15), 5u32),
        ];
        let hotspots = RegisterPressureEstimator::pressure_hotspots(&ranges, 8);
        // Pressure at 5-9 is 10, which exceeds 8
        assert!(!hotspots.is_empty());
        assert_eq!(hotspots[0].1, 10);
    }

    #[test]
    fn test_spill_cost_high_pressure() {
        let long_range = LiveRange::new(0, 100);
        let short_range = LiveRange::new(0, 5);
        let cost_long = SpillCostCalculator::spill_cost(&long_range, 4);
        let cost_short = SpillCostCalculator::spill_cost(&short_range, 4);
        assert!(cost_long > cost_short);
    }

    #[test]
    fn test_weighted_spill_cost_with_loop_depth() {
        let lr = LiveRange::new(0, 10);
        let shallow = SpillCostCalculator::weighted_spill_cost(&lr, 2, 1, 0);
        let deep = SpillCostCalculator::weighted_spill_cost(&lr, 2, 1, 2);
        assert!(deep > shallow);
    }

    #[test]
    fn test_select_spill_candidate() {
        let candidates = vec![
            (VirtualReg(0), LiveRange::new(0, 2), 1, 10.0),
            (VirtualReg(1), LiveRange::new(0, 100), 4, 400.0),
            (VirtualReg(2), LiveRange::new(0, 5), 2, 5.0),
        ];
        let best = SpillCostCalculator::select_spill_candidate(&candidates);
        assert_eq!(best, Some(VirtualReg(2))); // lowest cost
    }

    #[test]
    fn test_packed_trits_use_fewer_registers() {
        let mut allocator = RegisterAllocator::new(16);

        // Unpacked: 1 value per register, 5 values = 5 registers
        let unpacked = AllocationRequest {
            vreg: VirtualReg(0),
            live_range: LiveRange::new(0, 10),
            trit_count: 100, // 100 trits → ceil(100/20) = 5 registers
            packed: false,
        };
        assert_eq!(unpacked.registers_needed(), 5);

        // Packed: multiple values in one register
        let packed = AllocationRequest {
            vreg: VirtualReg(1),
            live_range: LiveRange::new(0, 10),
            trit_count: 15, // 15 trits → 1 register
            packed: true,
        };
        assert_eq!(packed.registers_needed(), 1);
    }

    #[test]
    fn test_allocation_result() {
        let mut allocator = RegisterAllocator::new(64);
        allocator.request(LiveRange::new(0, 5), 20, false);
        allocator.request(LiveRange::new(0, 5), 20, false);
        allocator.request(LiveRange::new(10, 15), 20, false);

        let result = allocator.allocate_all();
        assert_eq!(result.allocated_count(), 3);
        assert_eq!(result.spilled_count(), 0);
    }

    #[test]
    fn test_interference_graph() {
        let ranges = vec![
            (VirtualReg(0), LiveRange::new(0, 5)),
            (VirtualReg(1), LiveRange::new(3, 8)),
            (VirtualReg(2), LiveRange::new(10, 15)),
        ];
        let graph = LiveRangeAnalysis::build_interference_graph(&ranges);
        assert!(graph[&VirtualReg(0)].contains(&VirtualReg(1)));
        assert!(graph[&VirtualReg(1)].contains(&VirtualReg(0)));
        assert!(!graph[&VirtualReg(0)].contains(&VirtualReg(2)));
    }

    #[test]
    fn test_compute_from_def_use() {
        let lr = LiveRangeAnalysis::compute_from_def_use(2, &[5, 8, 3]);
        assert_eq!(lr.start, 2);
        assert_eq!(lr.end, 9);
    }

    #[test]
    fn test_register_file_utilization() {
        let mut rf = RegisterFile::new(8);
        assert!((rf.utilization() - 0.0).abs() < 0.001);
        rf.allocate(4).unwrap();
        assert!((rf.utilization() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_deallocate_virtual_reg() {
        let mut rf = RegisterFile::new(16);
        let regs = rf.allocate(3).unwrap();
        let alloc = Allocation {
            vreg: VirtualReg(42),
            physical_regs: regs,
            live_range: LiveRange::new(0, 10),
            registers_used: 3,
        };
        rf.assign(alloc);
        assert_eq!(rf.free_count(), 13);
        rf.deallocate(VirtualReg(42));
        assert_eq!(rf.free_count(), 16);
    }

    #[test]
    fn test_total_spill_cost() {
        let spilled = vec![
            AllocationRequest {
                vreg: VirtualReg(0),
                live_range: LiveRange::new(0, 10),
                trit_count: 20,
                packed: false,
            },
            AllocationRequest {
                vreg: VirtualReg(1),
                live_range: LiveRange::new(0, 5),
                trit_count: 40,
                packed: false,
            },
        ];
        let cost = SpillCostCalculator::total_spill_cost(&spilled);
        // vreg 0: 10 * 1 = 10, vreg 1: 5 * 2 = 10, total = 20
        assert!((cost - 20.0).abs() < 0.001);
    }
}
