/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToLinalg/StridedAxisCoalescing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include <functional>

namespace StridedAxisCoalescing {

using namespace mlir;
using namespace triton;

// Detects the FLA per-head strided base `base + (pid % S)` produced by
// splitting the H axis (the contiguous axis folded onto the grid). Returns the
// matching AddPtrOp, or a null AddPtrOp if `base` is not such an ih-split ptr.
static triton::AddPtrOp findIhAddPtr(Value base, int64_t S) {
    Value src = base;
    while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
        if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
        if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
            APInt cC;
            if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                std::abs(cC.getSExtValue()) == S) {
                Value lhs = rem.getLhs();
                while (true) {
                    if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                    if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                    break;
                }
                if (lhs.getDefiningOp<triton::GetProgramIdOp>()) return addptr;
            }
        }
        src = addptr.getPtr();
    }
    return triton::AddPtrOp();
}

// Mirror of findIhAddPtr that returns the program_id axis driving the ih split
// (i.e. the grid dim the host launcher must divide by S), or -1 if `base` is
// not such an ih-split ptr. Whenever findIhAddPtr succeeds this does too.
static int32_t findIhAxis(Value base, int64_t S) {
    Value src = base;
    while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
        if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
        if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
            APInt cC;
            if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                std::abs(cC.getSExtValue()) == S) {
                Value lhs = rem.getLhs();
                while (true) {
                    if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                    if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                    break;
                }
                if (auto pid = lhs.getDefiningOp<triton::GetProgramIdOp>())
                    return pid.getAxisAsInt();
            }
        }
        src = addptr.getPtr();
    }
    return -1;
}

// Returns the i_h value `pid % S` (the per-head index feeding the ih split), or
// null. Mirror of findIhAddPtr but yields the RemSIOp result itself, used to
// check whether i_h also feeds a per-head scalar load (which coalescing cannot
// lane-expand -- see the correctness guard in rewriteStridedAxisCoalesce).
static Value findIhRem(Value base, int64_t S) {
    Value src = base;
    while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
        if (isa<RankedTensorType>(addptr.getPtr().getType())) break;
        if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
            APInt cC;
            if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
                std::abs(cC.getSExtValue()) == S)
                return rem.getResult();
        }
        src = addptr.getPtr();
    }
    return Value();
}

static Value build2DBlockPtr(IRRewriter &rw, triton::MakeTensorPtrOp m1d,
                             int64_t S, int64_t BT) {
    triton::AddPtrOp ih = findIhAddPtr(m1d.getBase(), S);
    if (!ih) return Value();
    auto loc = m1d.getLoc();
    rw.setInsertionPoint(m1d);
    Value newBase = ih.getPtr();
    Value cH = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
    Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
    Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
    SmallVector<Value, 2> shape{m1d.getShape()[0], cH};
    SmallVector<Value, 2> strides{m1d.getStrides()[0], c1};
    SmallVector<Value, 2> offsets{m1d.getOffsets()[0], c0};
    SmallVector<int32_t, 2> blockShape{static_cast<int32_t>(BT), static_cast<int32_t>(S)};
    SmallVector<int32_t, 2> order{1, 0};
    auto p = rw.create<triton::MakeTensorPtrOp>(loc, newBase, shape, strides,
                                                offsets, blockShape, order);
    return p.getResult();
}

// Lift a rank-1 tensor type tensor<BTxe> to tensor<BTxSxe> (append the folded
// H axis as the inner lane). Scalars / non-rank-1 types pass through unchanged.
static Type lift2D(Type t, int64_t S) {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt || rt.getRank() != 1) return t;
    return RankedTensorType::get({rt.getShape()[0], S}, rt.getElementType());
}

static Type lift2DColInner(Type t, int64_t S) {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt || rt.getRank() != 1) return t;
    return RankedTensorType::get({S, rt.getShape()[0]}, rt.getElementType());
}

// An op is safe to 2D-ify (lane-parallel over the appended H axis) iff every
// lane s computes independently: a pure elementwise arith/math op, a cast, a
// splat, or a scan/reduce ALONG THE T axis (axis 0). On the 2D tile [BT,S] a
// T-axis reduce is per-lane (the S lanes stay independent, output [S]) and a
// T-axis scan likewise, so both lift directly -- no need to pre-collapse the
// reverse-cumsum idiom into a single scan. Ops that mix or move the lane
// (transpose, tt.dot, reshape, reduce/scan along the lane) are NOT here, so the
// caller bails and keeps the original (indirect) path.
static bool is2DSafe(Operation *op) {
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
            arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::CmpFOp, arith::SelectOp,
            arith::ExtFOp, arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp,
            arith::FPToSIOp, arith::FPToUIOp, arith::CmpIOp, arith::AndIOp,
            arith::OrIOp>(op))
        return true;
    // Every math dialect op (exp/log/sqrt/tanh/erf/...) is a per-lane scalar
    // elementwise map -> 2D-safe. This covers gate activations expressed as
    // tensor math (softplus = log(1+exp), silu, gelu, ...) without enumerating
    // each op, so such cumsum kernels coalesce without per-idiom pattern match.
    if (isa<math::MathDialect>(op->getDialect()))
        return true;
    if (isa<triton::SplatOp>(op)) return true;
    if (auto scan = dyn_cast<triton::ScanOp>(op))
        return scan.getAxis() == 0 && scan->getNumResults() == 1;
    if (auto reduce = dyn_cast<triton::ReduceOp>(op))
        return reduce.getAxis() == 0 && reduce->getNumResults() == 1;
    return false;
}

static void rewriteBlockPtrStridedAxisCoalesce(ModuleOp moduleOp) {
    IRRewriter rw(moduleOp.getContext());

    // Collect the strided ih-base 1D loads (seeds). All must share one stride S
    // (the folded H axis); BT is the per-chunk tile length.
    SmallVector<triton::LoadOp> seeds;
    int64_t S = 0, BT = 0;
    moduleOp.walk([&](triton::LoadOp l) {
        auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m) return;
        auto rt = dyn_cast<RankedTensorType>(l.getResult().getType());
        if (!rt || rt.getRank() != 1) return;
        auto strides = m.getStrides();
        if (strides.empty()) return;
        APInt sC;
        if (!matchPattern(strides.back(), m_ConstantInt(&sC))) return;
        int64_t s = std::abs(sC.getSExtValue());
        if (s <= 1) return;
        if (!findIhAddPtr(m.getBase(), s)) return;
        if (S == 0) { S = s; BT = rt.getShape()[0]; }
        if (s != S) return;
        seeds.push_back(l);
    });
    if (seeds.empty()) return;

    // The grid axis the launcher will divide by S (the pid feeding `pid % S`).
    int32_t coalesceAxis = -1;
    if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>())
        coalesceAxis = findIhAxis(m0.getBase(), S);
    if (coalesceAxis < 0) return;  // cannot identify the axis -> do not coalesce

    // Full TA path: the launcher divides grid[coalesceAxis] by S, so the
    // kernel-visible num_programs(coalesceAxis) becomes grid/S. If the kernel
    // reads it, coalescing would change that value -> wrong results. Bail (the
    // kernel keeps its original, correct, uncoalesced path).
    bool readsAxisNumPrograms = false;
    moduleOp.walk([&](triton::GetNumProgramsOp np) {
        if (np.getAxisAsInt() == coalesceAxis) readsAxisNumPrograms = true;
    });
    if (readsAxisNumPrograms) return;

    // The H axis is folded into the inner lane, so every per-head value must be
    // expanded across the S lanes. Block-ptr loads are lane-expanded by
    // build2DBlockPtr; a per-head SCALAR load (A_log[i_h] / dt_bias[i_h] in
    // gdn-style gates) is collected here and later lifted to an [S] per-lane
    // vector load (lane s -> base[s], matching the folded i_h = s). The scalar
    // chain on top of it (exp/neg/...) is lifted to [S] by get2D, and the splat
    // that feeds it into the tile becomes an [S]->[BT,S] broadcast. Bail only if
    // i_h reaches something we cannot lane-expand: a scalar load feeding an
    // address (indirect gather), or a tensor load not via make_tensor_ptr.
    SmallVector<triton::LoadOp> headLoads;
    if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (Value ihRem = findIhRem(m0.getBase(), S)) {
            SmallVector<Operation *> wl2(ihRem.getUsers().begin(),
                                         ihRem.getUsers().end());
            DenseSet<Operation *> seen2;
            while (!wl2.empty()) {
                Operation *u = wl2.pop_back_val();
                if (!seen2.insert(u).second) continue;
                if (isa<triton::MakeTensorPtrOp>(u)) continue;  // block-ptr path: fine
                if (auto ld = dyn_cast<triton::LoadOp>(u)) {
                    // per-head scalar load: must be scalar and must not itself feed
                    // an address (indirect gather). Then it is liftable to [S].
                    if (isa<RankedTensorType>(ld.getResult().getType())) return;
                    for (Operation *ru : ld.getResult().getUsers())
                        if (isa<triton::AddPtrOp>(ru)) return;  // indirect -> bail
                    headLoads.push_back(ld);
                    continue;
                }
                if (isa<triton::AddPtrOp, arith::ExtSIOp, arith::TruncIOp,
                        arith::RemSIOp, arith::AddIOp, arith::MulIOp>(u))
                    for (Operation *uu : u->getResult(0).getUsers())
                        wl2.push_back(uu);
            }
        }
    }

    // Discover the load->store subgraph by forward reachability from the seeds.
    // Every op on the way must be 2D-safe (elementwise / cast / splat / T-axis
    // scan or reduce); stores are the sinks. A T-axis reduce is kept as-is and
    // lifted in place (no idiom-specific pre-collapse) -- see is2DSafe.
    // Any unsafe op (or a value escaping to one) aborts the whole rewrite.
    DenseSet<Operation *> region;
    SmallVector<triton::StoreOp> sinks;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> wl;
    for (auto s : seeds)
        for (Operation *u : s.getResult().getUsers()) wl.push_back(u);
    while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!visited.insert(op).second) continue;
        if (auto st = dyn_cast<triton::StoreOp>(op)) { sinks.push_back(st); continue; }
        if (!is2DSafe(op)) return;  // bail: unsafe consumer in the chain
        region.insert(op);
        for (Value r : op->getResults())
            for (Operation *u : r.getUsers()) wl.push_back(u);
    }
    if (sinks.empty()) return;

    // Every sink store must also be a matching 1D stride-S ih-base block ptr.
    for (auto st : sinks) {
        auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        if (!m || !findIhAddPtr(m.getBase(), S)) return;
        auto os = m.getStrides();
        if (os.empty()) return;
        APInt soC;
        if (!matchPattern(os.back(), m_ConstantInt(&soC)) ||
            std::abs(soC.getSExtValue()) != S)
            return;
    }

    // Map each 1D value to its 2D counterpart, materializing splats/constants
    // on demand. Returns null to signal an un-liftable operand (bail).
    DenseMap<Value, Value> vmap;
    std::function<Value(Value)> get2D = [&](Value v) -> Value {
        auto it = vmap.find(v);
        if (it != vmap.end()) return it->second;
        if (!isa<RankedTensorType>(v.getType())) {
            // Scalar: if it derives from a per-head load (whose [S] lane vector is
            // already in vmap), lift the elementwise scalar chain to [S] so each
            // lane gets its own value. i_h-independent scalars stay scalar (they
            // splat uniformly across lanes, which is correct).
            Operation *def = v.getDefiningOp();
            bool liftable = def && (isa<math::MathDialect>(def->getDialect()) ||
                isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                    arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
                    arith::MaxNumFOp, arith::MinNumFOp, arith::ExtFOp,
                    arith::TruncFOp>(def));
            if (liftable) {
                SmallVector<Value> ops2;
                bool anyLane = false;
                for (Value o : def->getOperands()) {
                    Value n = get2D(o);
                    if (!n) return Value();
                    if (isa<RankedTensorType>(n.getType())) anyLane = true;
                    ops2.push_back(n);
                }
                if (anyLane) {
                    OpBuilder::InsertionGuard g(rw);
                    rw.setInsertionPointAfter(def);
                    for (Value &o : ops2)
                        if (!isa<RankedTensorType>(o.getType()))
                            o = rw.create<triton::SplatOp>(
                                def->getLoc(), RankedTensorType::get({S}, o.getType()), o);
                    OperationState st(def->getLoc(), def->getName());
                    st.addOperands(ops2);
                    st.addAttributes(def->getAttrs());
                    for (Value r : def->getResults())
                        st.addTypes(RankedTensorType::get({S}, r.getType()));
                    Operation *nu = rw.create(st);
                    vmap[v] = nu->getResult(0);
                    return nu->getResult(0);
                }
            }
            return v;  // i_h-independent scalar, splats uniformly
        }
        // Save/restore the insertion point: the materializers below move it next
        // to the original splat/constant (which may sit at the top of the func).
        // Without this the caller's `rw.setInsertionPoint(op)` would be clobbered
        // and the rebuilt op emitted before its operands -> dominance violation.
        OpBuilder::InsertionGuard guard(rw);
        if (auto sp = v.getDefiningOp<triton::SplatOp>()) {
            Value src2 = get2D(sp.getSrc());
            if (!src2) return Value();
            rw.setInsertionPointAfter(sp);
            Value n;
            if (isa<RankedTensorType>(src2.getType())) {
                // src2 is the per-lane reduce result [S] (the 1D scalar source
                // became a vector once the reduce was 2D-ified). Broadcast it
                // across T: [S] -> expand_dims(0) -> [1,S] -> broadcast [BT,S].
                Value ex = rw.create<triton::ExpandDimsOp>(sp.getLoc(), src2, 0);
                n = rw.create<triton::BroadcastOp>(sp.getLoc(),
                                                   lift2D(sp.getType(), S), ex);
            } else {
                // True scalar splat: lift to a 2D splat over [BT,S].
                n = rw.create<triton::SplatOp>(sp.getLoc(),
                                               lift2D(sp.getType(), S), src2);
            }
            vmap[v] = n;
            return n;
        }
        if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            if (auto dea = dyn_cast<DenseElementsAttr>(c.getValue())) {
                if (dea.isSplat()) {
                    auto nt = cast<RankedTensorType>(lift2D(c.getType(), S));
                    rw.setInsertionPointAfter(c);
                    Value n = rw.create<arith::ConstantOp>(
                        c.getLoc(), nt,
                        DenseElementsAttr::get(nt, dea.getSplatValue<Attribute>()));
                    vmap[v] = n;
                    return n;
                }
            }
        }
        return Value();
    };

    // Build 2D loads for the seeds.
    for (auto l : seeds) {
        auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        Value p2 = build2DBlockPtr(rw, m, S, BT);
        if (!p2) return;
        rw.setInsertionPoint(l);
        auto nl = rw.create<triton::LoadOp>(
            l.getLoc(), p2, ArrayRef<int32_t>{0, 1}, l.getPadding(),
            l.getCache(), l.getEvict(), l.getIsVolatile());
        vmap[l.getResult()] = nl.getResult();
    }

    // Lift each per-head scalar load to an [S] per-lane vector load: base[0:S]
    // (lane s = base[s], matching the folded i_h = s). The offset must be exactly
    // i_h = pid % S so that lane s maps to base[s]; otherwise bail.
    for (auto ld : headLoads) {
        auto ap = ld.getPtr().getDefiningOp<triton::AddPtrOp>();
        if (!ap) return;
        Value off = ap.getOffset();
        while (auto e = off.getDefiningOp<arith::ExtSIOp>()) off = e.getIn();
        while (auto t = off.getDefiningOp<arith::TruncIOp>()) off = t.getIn();
        if (!off.getDefiningOp<arith::RemSIOp>()) return;  // offset not pure i_h
        Value base = ap.getPtr();
        Type elemTy = ld.getResult().getType();
        rw.setInsertionPoint(ld);
        auto loc = ld.getLoc();
        Value cS = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
        Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
        Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
        SmallVector<Value, 1> shape{cS}, strides{c1}, offsets{c0};
        SmallVector<int32_t, 1> blockShape{static_cast<int32_t>(S)}, order{0};
        auto p = rw.create<triton::MakeTensorPtrOp>(loc, base, shape, strides,
                                                    offsets, blockShape, order);
        auto vl = rw.create<triton::LoadOp>(
            loc, p.getResult(), ArrayRef<int32_t>{0}, triton::PaddingOption::PAD_ZERO,
            triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
        (void)elemTy;
        vmap[ld.getResult()] = vl.getResult();
    }

    // Rebuild the region ops in IR (topological) order as 2D.
    SmallVector<Operation *> ordered;
    moduleOp.walk([&](Operation *op) { if (region.count(op)) ordered.push_back(op); });
    for (Operation *op : ordered) {
        rw.setInsertionPoint(op);
        if (auto scan = dyn_cast<triton::ScanOp>(op)) {
            Value in = get2D(scan.getOperand(0));
            if (!in) return;
            auto ns = rw.create<triton::ScanOp>(scan.getLoc(), ValueRange{in},
                                                static_cast<int>(scan.getAxis()),
                                                scan.getReverse());
            rw.cloneRegionBefore(scan.getCombineOp(), ns.getCombineOp(),
                                 ns.getCombineOp().end());
            vmap[scan->getResult(0)] = ns->getResult(0);
            continue;
        }
        if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
            Value in = get2D(reduce.getOperand(0));
            if (!in) return;
            // T-axis reduce on the 2D tile [BT,S] -> [S]: one independent
            // reduction per lane (the S lanes do not mix). The 1D result was a
            // scalar; its 2D counterpart is the per-lane vector [S], which a
            // downstream splat turns into an expand_dims+broadcast (see get2D).
            auto nr = rw.create<triton::ReduceOp>(reduce.getLoc(), ValueRange{in},
                                                  static_cast<int>(reduce.getAxis()));
            rw.cloneRegionBefore(reduce.getCombineOp(), nr.getCombineOp(),
                                 nr.getCombineOp().end());
            vmap[reduce->getResult(0)] = nr->getResult(0);
            continue;
        }
        // Splats are materialized on demand by get2D (which lifts a scalar splat
        // to a 2D splat, and a per-lane reduce result [S] to expand_dims +
        // broadcast). Skip here so it is not rebuilt as an invalid 2D splat.
        if (isa<triton::SplatOp>(op)) continue;
        SmallVector<Value> operands;
        for (Value o : op->getOperands()) {
            Value n = get2D(o);
            if (!n) return;
            operands.push_back(n);
        }
        OperationState st(op->getLoc(), op->getName());
        st.addOperands(operands);
        st.addAttributes(op->getAttrs());
        for (Value r : op->getResults()) st.addTypes(lift2D(r.getType(), S));
        Operation *nu = rw.create(st);
        for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
            vmap[oldR] = newR;
    }

    // Build the 2D stores.
    for (auto st : sinks) {
        Value val = get2D(st.getValue());
        if (!val) return;
        auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
        Value p2 = build2DBlockPtr(rw, m, S, BT);
        if (!p2) return;
        rw.setInsertionPoint(st);
        rw.create<triton::StoreOp>(st.getLoc(), p2, val, ArrayRef<int32_t>{0, 1},
                                   st.getCache(), st.getEvict());
    }

    // i_b = divsi(get_program_id, S): with the H axis folded into the inner
    // tile the per-instance i_b becomes the raw program id. Redirect it (this
    // also fixes the new 2D block ptr bases, which reuse i_b).
    if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
        if (triton::AddPtrOp ihA = findIhAddPtr(m0.getBase(), S)) {
            if (auto rem = ihA.getOffset().getDefiningOp<arith::RemSIOp>()) {
                Value lhs = rem.getLhs();
                while (true) {
                    if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) { lhs = e.getIn(); continue; }
                    if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) { lhs = t.getIn(); continue; }
                    break;
                }
                SmallVector<arith::DivSIOp, 2> divs;
                for (Operation *u : lhs.getUsers())
                    if (auto dv = dyn_cast<arith::DivSIOp>(u)) {
                        APInt dC;
                        if (dv.getLhs() == lhs &&
                            matchPattern(dv.getRhs(), m_ConstantInt(&dC)) &&
                            std::abs(dC.getSExtValue()) == S)
                            divs.push_back(dv);
                    }
                for (auto dv : divs) rw.replaceAllUsesWith(dv.getResult(), lhs);
            }
        }
    }

    // Erase the original chain (sinks, then region in reverse order, then seeds).
    for (auto st : sinks) rw.eraseOp(st);
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) rw.eraseOp(*it);
    for (auto l : seeds) rw.eraseOp(l);

    auto i32t = IntegerType::get(moduleOp.getContext(), 32);
    moduleOp->setAttr("hacc.coalesce_factor", IntegerAttr::get(i32t, S));
    moduleOp->setAttr("hacc.coalesce_axis", IntegerAttr::get(i32t, coalesceAxis));
}

// Addptr-addressed kernels do not have a structured block pointer in the input
// IR. This side path recognizes the flattened launch/address shape directly:
//   output_row = pid / cdiv(D, BT), group = pid % cdiv(D, BT)
//   batch = output_row / S, h = output_row % S
//   ptr = base + begin + h * D + (range(0, BT) + group * BT)
// and rebuilds the load/compute/store chain as a [S,BT] tile. Keeping BT inner
// preserves contiguous GM accesses on the column dimension.
struct AddPtrAccess {
    enum class RowKind { Jagged, Dense };

    Value outputRow;
    Value batch;
    Value lane;
    Value d;
    Value cols;
    Value colMask;
    Value rowMask;
    Value begin;
    Value base;
    RowKind rowKind = RowKind::Jagged;
};

struct AddPtrSeed {
    triton::LoadOp load;
    int64_t S = 0;
    int64_t BT = 0;
    int32_t coalesceAxis = -1;
    AddPtrAccess access;
};

struct AddPtrStore {
    triton::StoreOp store;
    AddPtrAccess access;
    Value base;
    Value baseOffset;
    bool includeBatch = true;
};

static bool matchFlattenedRowAndLane(triton::GetProgramIdOp pidOp,
                                     Value &gridDim, Value &outputRow,
                                     Value &group, Value &batch, Value &lane,
                                     int64_t &S) {
    for (Operation *user : pidOp.getResult().getUsers()) {
        auto div = dyn_cast<arith::DivSIOp>(user);
        if (!div || div.getLhs() != pidOp.getResult()) continue;

        Value candidateGridDim = div.getRhs();
        Value candidateGroup;
        for (Operation *maybeRemUser : pidOp.getResult().getUsers()) {
            auto rem = dyn_cast<arith::RemSIOp>(maybeRemUser);
            if (rem && rem.getLhs() == pidOp.getResult() &&
                rem.getRhs() == candidateGridDim) {
                candidateGroup = rem.getResult();
                break;
            }
        }
        if (!candidateGroup) continue;

        for (Operation *rowUser : div.getResult().getUsers()) {
            auto batchDiv = dyn_cast<arith::DivSIOp>(rowUser);
            if (!batchDiv || batchDiv.getLhs() != div.getResult()) continue;
            auto candidateS = getConstantIntValue(batchDiv.getRhs());
            if (!candidateS || *candidateS <= 1) continue;

            Value candidateLane;
            for (Operation *maybeLaneUser : div.getResult().getUsers()) {
                auto rem = dyn_cast<arith::RemSIOp>(maybeLaneUser);
                auto remS = rem ? getConstantIntValue(rem.getRhs()) : std::nullopt;
                if (rem && rem.getLhs() == div.getResult() && remS &&
                    *remS == *candidateS) {
                    candidateLane = rem.getResult();
                    break;
                }
            }

            gridDim = candidateGridDim;
            outputRow = div.getResult();
            group = candidateGroup;
            batch = batchDiv.getResult();
            lane = candidateLane;
            S = *candidateS;
            return true;
        }
    }
    return false;
}

static bool matchColumnOffsets(Value cols, Value group, Value &d,
                               Value &colMask, int64_t &BT) {
    auto add = cols.getDefiningOp<arith::AddIOp>();
    if (!add) return false;

    triton::MakeRangeOp range;
    Value colBase;
    if ((range = add.getLhs().getDefiningOp<triton::MakeRangeOp>())) {
        if (auto splat = add.getRhs().getDefiningOp<triton::SplatOp>())
            colBase = splat.getSrc();
    } else if ((range = add.getRhs().getDefiningOp<triton::MakeRangeOp>())) {
        if (auto splat = add.getLhs().getDefiningOp<triton::SplatOp>())
            colBase = splat.getSrc();
    }
    if (!range || range.getStart() != 0 || !colBase) return false;

    BT = range.getEnd();
    auto colBaseMul = colBase.getDefiningOp<arith::MulIOp>();
    if (BT <= 1 || !colBaseMul) return false;
    auto lhsConst = getConstantIntValue(colBaseMul.getLhs());
    auto rhsConst = getConstantIntValue(colBaseMul.getRhs());
    if (!((colBaseMul.getLhs() == group && rhsConst && *rhsConst == BT) ||
          (colBaseMul.getRhs() == group && lhsConst && *lhsConst == BT)))
        return false;

    for (Operation *user : cols.getUsers()) {
        auto cmp = dyn_cast<arith::CmpIOp>(user);
        if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt ||
            cmp.getLhs() != cols)
            continue;
        auto splat = cmp.getRhs().getDefiningOp<triton::SplatOp>();
        if (!splat) continue;
        d = splat.getSrc();
        colMask = cmp.getResult();
        return true;
    }
    return false;
}

static bool matchBeginEnd(Value batch, Value &begin) {
    for (Operation *user : batch.getUsers()) {
        auto beginPtr = dyn_cast<triton::AddPtrOp>(user);
        if (!beginPtr || beginPtr.getOffset() != batch) continue;

        triton::LoadOp beginLoad;
        for (Operation *ptrUser : beginPtr.getResult().getUsers()) {
            beginLoad = dyn_cast<triton::LoadOp>(ptrUser);
            if (beginLoad) break;
        }
        if (!beginLoad) continue;

        Value base = beginPtr.getPtr();
        for (Operation *batchUser : batch.getUsers()) {
            auto add = dyn_cast<arith::AddIOp>(batchUser);
            if (!add ||
                !((add.getLhs() == batch && getConstantIntValue(add.getRhs()) &&
                   *getConstantIntValue(add.getRhs()) == 1) ||
                  (add.getRhs() == batch && getConstantIntValue(add.getLhs()) &&
                   *getConstantIntValue(add.getLhs()) == 1)))
                continue;
            for (Operation *nextUser : add.getResult().getUsers()) {
                auto endPtr = dyn_cast<triton::AddPtrOp>(nextUser);
                if (!endPtr || endPtr.getPtr() != base ||
                    endPtr.getOffset() != add.getResult())
                    continue;
                for (Operation *ptrUser : endPtr.getResult().getUsers()) {
                    auto endLoad = dyn_cast<triton::LoadOp>(ptrUser);
                    if (!endLoad) continue;
                    begin = beginLoad.getResult();
                    return true;
                }
            }
        }
    }
    return false;
}

static bool matchJaggedRowPtr(Value rowPtr, Value begin, Value lane, Value d,
                              Value &base) {
    auto addPtr = rowPtr.getDefiningOp<triton::AddPtrOp>();
    if (!addPtr) return false;
    auto add = addPtr.getOffset().getDefiningOp<arith::AddIOp>();
    if (!add) return false;

    Value laneOffset = add.getLhs() == begin ? add.getRhs() : add.getLhs();
    while (true) {
        if (auto e = laneOffset.getDefiningOp<arith::ExtSIOp>()) {
            laneOffset = e.getIn();
            continue;
        }
        if (auto t = laneOffset.getDefiningOp<arith::TruncIOp>()) {
            laneOffset = t.getIn();
            continue;
        }
        break;
    }
    auto laneMul = laneOffset.getDefiningOp<arith::MulIOp>();
    if (!((add.getLhs() == begin || add.getRhs() == begin) && laneMul &&
          ((laneMul.getLhs() == lane && laneMul.getRhs() == d) ||
           (laneMul.getLhs() == d && laneMul.getRhs() == lane))))
        return false;

    base = addPtr.getPtr();
    return true;
}

static bool matchDenseRowPtr(Value rowPtr, Value outputRow, Value d,
                             Value &base) {
    auto addPtr = rowPtr.getDefiningOp<triton::AddPtrOp>();
    auto mul = addPtr ? addPtr.getOffset().getDefiningOp<arith::MulIOp>()
                      : arith::MulIOp();
    if (!addPtr || !mul ||
        !((mul.getLhs() == outputRow && mul.getRhs() == d) ||
          (mul.getLhs() == d && mul.getRhs() == outputRow)))
        return false;
    base = addPtr.getPtr();
    return true;
}

static bool matchLoadMask(Value mask, Value colMask, Value &rowMask) {
    if (mask == colMask) return true;
    auto andOp = mask.getDefiningOp<arith::AndIOp>();
    if (!andOp) return false;
    if (andOp.getLhs() == colMask) {
        if (auto splat = andOp.getRhs().getDefiningOp<triton::SplatOp>())
            rowMask = splat.getSrc();
        return static_cast<bool>(rowMask);
    }
    if (andOp.getRhs() == colMask) {
        if (auto splat = andOp.getLhs().getDefiningOp<triton::SplatOp>())
            rowMask = splat.getSrc();
        return static_cast<bool>(rowMask);
    }
    return false;
}

static bool matchAddPtrLoad(triton::LoadOp load, AddPtrSeed &seed) {
    auto vectorType = dyn_cast<RankedTensorType>(load.getType());
    if (!vectorType || vectorType.getRank() != 1) return false;
    auto addPtr = load.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!addPtr) return false;
    Value rowPtr;
    if (auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>())
        rowPtr = splat.getSrc();
    if (!rowPtr) return false;

    auto func = load->getParentOfType<triton::FuncOp>();
    if (!func) return false;
    Block &body = func.getBody().front();
    for (Operation &op : body) {
        auto pid = dyn_cast<triton::GetProgramIdOp>(op);
        if (!pid) continue;

        AddPtrAccess candidate;
        Value gridDim;
        Value group;
        int64_t S = 0;
        int64_t BT = 0;
        if (!matchFlattenedRowAndLane(pid, gridDim, candidate.outputRow,
                                      group, candidate.batch, candidate.lane,
                                      S))
            continue;
        if (!matchColumnOffsets(addPtr.getOffset(), group, candidate.d,
                                candidate.colMask, BT))
            continue;
        auto gridDiv = gridDim.getDefiningOp<arith::DivSIOp>();
        auto gridBiasAdd = gridDiv
                               ? gridDiv.getLhs().getDefiningOp<arith::AddIOp>()
                               : arith::AddIOp();
        int64_t bias = BT - 1;
        bool isCdiv = false;
        if (gridDiv && gridBiasAdd) {
            auto gridDivisor = getConstantIntValue(gridDiv.getRhs());
            auto lhsBias = getConstantIntValue(gridBiasAdd.getRhs());
            auto rhsBias = getConstantIntValue(gridBiasAdd.getLhs());
            isCdiv = gridDivisor && *gridDivisor == BT &&
                     ((gridBiasAdd.getLhs() == candidate.d && lhsBias &&
                       *lhsBias == bias) ||
                      (gridBiasAdd.getRhs() == candidate.d && rhsBias &&
                       *rhsBias == bias));
        }
        if (vectorType.getShape()[0] != BT || !isCdiv)
            continue;
        if (!load.getMask() ||
            !matchLoadMask(load.getMask(), candidate.colMask,
                           candidate.rowMask))
            continue;
        if (matchDenseRowPtr(rowPtr, candidate.outputRow, candidate.d,
                             candidate.base)) {
            candidate.rowKind = AddPtrAccess::RowKind::Dense;
        } else {
            if (!matchBeginEnd(candidate.batch, candidate.begin)) continue;
            if (!matchJaggedRowPtr(rowPtr, candidate.begin, candidate.lane,
                                   candidate.d, candidate.base))
                continue;
            candidate.rowKind = AddPtrAccess::RowKind::Jagged;
        }

        candidate.cols = addPtr.getOffset();
        seed.load = load;
        seed.S = S;
        seed.BT = BT;
        seed.coalesceAxis = pid.getAxisAsInt();
        seed.access = candidate;
        return true;
    }
    return false;
}

static bool matchAddPtrStore(triton::StoreOp store, ArrayRef<AddPtrSeed> seeds,
                             AddPtrStore &sink) {
    for (const AddPtrSeed &seed : seeds) {
        auto addPtr = store.getPtr().getDefiningOp<triton::AddPtrOp>();
        if (!addPtr || addPtr.getOffset() != seed.access.cols) continue;
        Value rowPtr;
        if (auto splat = addPtr.getPtr().getDefiningOp<triton::SplatOp>())
            rowPtr = splat.getSrc();
        if (!rowPtr || store.getMask() != seed.access.colMask) continue;

        Value base;
        Value baseOffset;
        bool includeBatch = true;
        if (!matchDenseRowPtr(rowPtr, seed.access.outputRow, seed.access.d,
                              base)) {
            if (!seed.access.begin ||
                !matchJaggedRowPtr(rowPtr, seed.access.begin,
                                   seed.access.lane, seed.access.d, base))
                continue;
            baseOffset = seed.access.begin;
            includeBatch = false;
        }

        sink.store = store;
        sink.access = seed.access;
        sink.base = base;
        sink.baseOffset = baseOffset;
        sink.includeBatch = includeBatch;
        return true;
    }
    return false;
}

static Value buildAddPtrOffsets2D(IRRewriter &rw, Location loc,
                                  const AddPtrAccess &access, int64_t S,
                                  int64_t BT, Value baseOffset,
                                  bool includeBatch) {
    Type i32Ty = rw.getI32Type();
    Type i64Ty = rw.getI64Type();
    auto hI32Ty = RankedTensorType::get({S}, i32Ty);
    auto hI64Ty = RankedTensorType::get({S}, i64Ty);
    auto tileI64Ty = RankedTensorType::get({S, BT}, i64Ty);

    Value cS = rw.create<arith::ConstantIntOp>(loc, S, 32);
    Value hs = rw.create<triton::MakeRangeOp>(loc, hI32Ty, 0, S);
    Value rowBase = hs;
    if (includeBatch) {
        Value batchS = rw.create<arith::MulIOp>(loc, access.batch, cS);
        Value batchSplat = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({S}, i32Ty), batchS);
        rowBase = rw.create<arith::AddIOp>(loc, batchSplat, hs);
    }
    Value dRows = rw.create<triton::SplatOp>(
        loc, RankedTensorType::get({S}, i32Ty), access.d);
    Value rowTimesD = rw.create<arith::MulIOp>(loc, rowBase, dRows);
    Value rowTimesD64 = rw.create<arith::ExtSIOp>(loc, hI64Ty, rowTimesD);
    Value expandedRows = rw.create<triton::ExpandDimsOp>(loc, rowTimesD64, 1);
    Value rowOffsets2d =
        rw.create<triton::BroadcastOp>(loc, tileI64Ty, expandedRows);

    auto colI64Ty = RankedTensorType::get({BT}, i64Ty);
    Value cols64 = rw.create<arith::ExtSIOp>(loc, colI64Ty, access.cols);
    Value expandedCols = rw.create<triton::ExpandDimsOp>(loc, cols64, 0);
    Value cols2d = rw.create<triton::BroadcastOp>(loc, tileI64Ty,
                                                  expandedCols);
    Value offsets = rw.create<arith::AddIOp>(loc, rowOffsets2d, cols2d);
    if (baseOffset) {
        Value base2d = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({S, BT}, i64Ty), baseOffset);
        offsets = rw.create<arith::AddIOp>(loc, base2d, offsets);
    }
    return offsets;
}

static Value buildAddPtrLoad2D(IRRewriter &rw, const AddPtrSeed &seed,
                               const std::function<Value(Value)> &get2D) {
    triton::LoadOp load = seed.load;
    rw.setInsertionPoint(load);
    Value other = load.getOther() ? get2D(load.getOther()) : Value();
    if (load.getOther() && !other) return Value();
    const AddPtrAccess &access = seed.access;
    Location loc = load.getLoc();
    auto tileType = cast<RankedTensorType>(lift2DColInner(load.getType(),
                                                          seed.S));
    auto tilePtrType =
        RankedTensorType::get({seed.S, seed.BT}, access.base.getType());
    auto tileI1Ty =
        RankedTensorType::get({seed.S, seed.BT}, rw.getI1Type());

    Value offsets = buildAddPtrOffsets2D(
        rw, loc, access, seed.S, seed.BT,
        access.rowKind == AddPtrAccess::RowKind::Jagged ? access.begin
                                                        : Value(),
        access.rowKind == AddPtrAccess::RowKind::Dense);
    Value base = rw.create<triton::SplatOp>(loc, tilePtrType, access.base);
    Value ptrs = rw.create<triton::AddPtrOp>(loc, tilePtrType, base, offsets);

    Value expandedColMask = rw.create<triton::ExpandDimsOp>(loc,
                                                            access.colMask, 0);
    Value colMask2d = rw.create<triton::BroadcastOp>(loc, tileI1Ty,
                                                     expandedColMask);
    Value mask = colMask2d;
    if (access.rowMask) {
        Value rowMask2d = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({seed.S, seed.BT}, rw.getI1Type()),
            access.rowMask);
        mask = rw.create<arith::AndIOp>(loc, colMask2d, rowMask2d);
    }
    if (!other) {
        Attribute zero = rw.getZeroAttr(tileType.getElementType());
        other = rw.create<arith::ConstantOp>(
            loc, DenseElementsAttr::get(tileType, zero));
    }
    return rw.create<triton::LoadOp>(
                 loc, ptrs, mask, other, ArrayRef<int32_t>{}, nullptr,
                 load.getCache(), load.getEvict(), load.getIsVolatile())
        .getResult();
}

static Value buildDenseLoad2D(IRRewriter &rw, triton::LoadOp load,
                              const AddPtrSeed &seed, Value base) {
    rw.setInsertionPoint(load);
    const AddPtrAccess &access = seed.access;
    Location loc = load.getLoc();
    Type elemTy = load.getType();
    Type i32Ty = rw.getI32Type();
    auto hI32Ty = RankedTensorType::get({seed.S}, i32Ty);
    auto hElemTy = RankedTensorType::get({seed.S}, elemTy);
    auto hPtrTy = RankedTensorType::get({seed.S}, base.getType());

    Value cS = rw.create<arith::ConstantIntOp>(loc, seed.S, 32);
    Value hs = rw.create<triton::MakeRangeOp>(loc, hI32Ty, 0, seed.S);
    Value batchS = rw.create<arith::MulIOp>(loc, access.batch, cS);
    Value batchSplat = rw.create<triton::SplatOp>(
        loc, RankedTensorType::get({seed.S}, i32Ty), batchS);
    Value offsets = rw.create<arith::AddIOp>(loc, batchSplat, hs);
    Value baseSplat = rw.create<triton::SplatOp>(loc, hPtrTy, base);
    Value ptrs = rw.create<triton::AddPtrOp>(loc, hPtrTy, baseSplat, offsets);
    Value mask = Value();
    if (access.rowMask)
        mask = rw.create<triton::SplatOp>(
            loc, RankedTensorType::get({seed.S}, rw.getI1Type()),
            access.rowMask);
    Attribute zero = rw.getZeroAttr(hElemTy.getElementType());
    Value other = rw.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(hElemTy, zero));
    return rw.create<triton::LoadOp>(
                 loc, ptrs, mask, other, ArrayRef<int32_t>{}, nullptr,
                 load.getCache(), load.getEvict(), load.getIsVolatile())
        .getResult();
}

static bool buildAddPtrStore2D(IRRewriter &rw, const AddPtrStore &sink,
                               Value value, int64_t S, int64_t BT) {
    triton::StoreOp store = sink.store;
    rw.setInsertionPoint(store);
    const AddPtrAccess &access = sink.access;
    Location loc = store.getLoc();
    auto tilePtrType = RankedTensorType::get({S, BT}, sink.base.getType());
    auto tileI1Ty = RankedTensorType::get({S, BT}, rw.getI1Type());
    Value offsets = buildAddPtrOffsets2D(rw, loc, access, S, BT,
                                         sink.baseOffset, sink.includeBatch);
    Value baseSplat = rw.create<triton::SplatOp>(loc, tilePtrType, sink.base);
    Value ptrs = rw.create<triton::AddPtrOp>(loc, tilePtrType, baseSplat,
                                             offsets);
    Value expandedMask = rw.create<triton::ExpandDimsOp>(loc, access.colMask,
                                                         0);
    Value mask = rw.create<triton::BroadcastOp>(loc, tileI1Ty, expandedMask);
    rw.create<triton::StoreOp>(loc, ptrs, value, mask);
    return true;
}

static void rewriteAddPtrStridedAxisCoalesce(ModuleOp moduleOp) {
    if (moduleOp->hasAttr("hacc.coalesce_factor"))
        return;

    AddPtrSeed rootSeed;
    bool foundRoot = false;
    moduleOp.walk([&](triton::LoadOp load) {
        if (foundRoot) return;
        foundRoot = matchAddPtrLoad(load, rootSeed);
    });
    if (!foundRoot) return;

    int64_t S = rootSeed.S, BT = rootSeed.BT;
    int32_t coalesceAxis = rootSeed.coalesceAxis;
    if (S <= 1 || BT <= 0 || coalesceAxis < 0) return;

    bool readsAxisNumPrograms = false;
    moduleOp.walk([&](triton::GetNumProgramsOp np) {
        if (np.getAxisAsInt() == coalesceAxis) readsAxisNumPrograms = true;
    });
    if (readsAxisNumPrograms) return;

    SmallVector<AddPtrSeed> seeds{rootSeed};
    moduleOp.walk([&](triton::LoadOp load) {
        if (load == rootSeed.load) return;
        AddPtrSeed seed;
        if (!matchAddPtrLoad(load, seed)) return;
        if (seed.S != S || seed.BT != BT || seed.coalesceAxis != coalesceAxis)
            return;
        seeds.push_back(seed);
    });

    DenseSet<Operation *> seedOps;
    for (AddPtrSeed seed : seeds)
        seedOps.insert(seed.load.getOperation());

    SmallVector<triton::LoadOp> denseLoads;
    DenseMap<Operation *, Value> denseBases;
    DenseMap<Operation *, unsigned> denseSeedIndex;
    DenseSet<Operation *> seenDenseLoads;
    for (unsigned i = 0; i < seeds.size(); ++i) {
        const AddPtrSeed &seed = seeds[i];
        moduleOp.walk([&](triton::LoadOp load) {
            if (seedOps.contains(load.getOperation())) return;
            if (isa<RankedTensorType>(load.getType())) return;
            Value ptr = load.getPtr();
            if (auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>()) {
                auto zero = getConstantIntValue(addPtr.getOffset());
                if (zero && *zero == 0)
                    ptr = addPtr.getPtr();
            }
            auto addPtr = ptr.getDefiningOp<triton::AddPtrOp>();
            if (!addPtr || addPtr.getOffset() != seed.access.outputRow) return;
            if (load.getMask() && seed.access.rowMask &&
                load.getMask() != seed.access.rowMask)
                return;
            if (!seenDenseLoads.insert(load.getOperation()).second) return;
            denseLoads.push_back(load);
            denseBases[load.getOperation()] = addPtr.getPtr();
            denseSeedIndex[load.getOperation()] = i;
        });
    }

    DenseSet<Operation *> region;
    SmallVector<triton::StoreOp> sinks;
    DenseSet<Operation *> visited;
    SmallVector<Operation *> wl;
    for (AddPtrSeed seed : seeds)
        for (Operation *user : seed.load.getResult().getUsers())
            wl.push_back(user);
    while (!wl.empty()) {
        Operation *op = wl.pop_back_val();
        if (!visited.insert(op).second) continue;
        if (auto st = dyn_cast<triton::StoreOp>(op)) {
            sinks.push_back(st);
            continue;
        }
        if (!is2DSafe(op)) return;
        region.insert(op);
        for (Value result : op->getResults())
            for (Operation *user : result.getUsers())
                wl.push_back(user);
    }
    if (sinks.empty()) return;

    SmallVector<AddPtrStore> coalescedStores;
    for (triton::StoreOp st : sinks) {
        AddPtrStore store;
        if (!matchAddPtrStore(st, seeds, store))
            return;
        coalescedStores.push_back(store);
    }

    IRRewriter rw(moduleOp.getContext());
    DenseMap<Value, Value> vmap;
    std::function<Value(Value)> get2D = [&](Value v) -> Value {
        auto it = vmap.find(v);
        if (it != vmap.end()) return it->second;
        if (!isa<RankedTensorType>(v.getType())) {
            Operation *def = v.getDefiningOp();
            bool liftable = def && (isa<math::MathDialect>(def->getDialect()) ||
                isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
                    arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp,
                    arith::MaxNumFOp, arith::MinNumFOp, arith::ExtFOp,
                    arith::TruncFOp>(def));
            if (liftable) {
                SmallVector<Value> ops2;
                bool anyLane = false;
                for (Value operand : def->getOperands()) {
                    Value n = get2D(operand);
                    if (!n) return Value();
                    if (isa<RankedTensorType>(n.getType())) anyLane = true;
                    ops2.push_back(n);
                }
                if (anyLane) {
                    OpBuilder::InsertionGuard guard(rw);
                    rw.setInsertionPointAfter(def);
                    for (Value &operand : ops2)
                        if (!isa<RankedTensorType>(operand.getType()))
                            operand = rw.create<triton::SplatOp>(
                                def->getLoc(),
                                RankedTensorType::get({S}, operand.getType()),
                                operand);
                    OperationState st(def->getLoc(), def->getName());
                    st.addOperands(ops2);
                    st.addAttributes(def->getAttrs());
                    for (Value result : def->getResults())
                        st.addTypes(RankedTensorType::get({S},
                                                          result.getType()));
                    Operation *nu = rw.create(st);
                    vmap[v] = nu->getResult(0);
                    return nu->getResult(0);
                }
            }
            return v;
        }

        OpBuilder::InsertionGuard guard(rw);
        if (auto sp = v.getDefiningOp<triton::SplatOp>()) {
            Value src2 = get2D(sp.getSrc());
            if (!src2) return Value();
            rw.setInsertionPointAfter(sp);
            Value n;
            if (isa<RankedTensorType>(src2.getType())) {
                Value ex = rw.create<triton::ExpandDimsOp>(sp.getLoc(), src2,
                                                           1);
                n = rw.create<triton::BroadcastOp>(
                    sp.getLoc(), lift2DColInner(sp.getType(), S), ex);
            } else {
                n = rw.create<triton::SplatOp>(
                    sp.getLoc(), lift2DColInner(sp.getType(), S), src2);
            }
            vmap[v] = n;
            return n;
        }
        if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
            if (auto dea = dyn_cast<DenseElementsAttr>(c.getValue())) {
                if (dea.isSplat()) {
                    auto nt = cast<RankedTensorType>(
                        lift2DColInner(c.getType(), S));
                    rw.setInsertionPointAfter(c);
                    Value n = rw.create<arith::ConstantOp>(
                        c.getLoc(), nt,
                        DenseElementsAttr::get(nt,
                                                dea.getSplatValue<Attribute>()));
                    vmap[v] = n;
                    return n;
                }
            }
        }
        if (Operation *def = v.getDefiningOp()) {
            if (is2DSafe(def) && def->getNumResults() == 1) {
                SmallVector<Value> operands;
                bool any2D = false;
                for (Value operand : def->getOperands()) {
                    Value n = get2D(operand);
                    if (!n) return Value();
                    if (isa<RankedTensorType>(n.getType())) any2D = true;
                    operands.push_back(n);
                }
                if (any2D) {
                    rw.setInsertionPointAfter(def);
                    OperationState st(def->getLoc(), def->getName());
                    st.addOperands(operands);
                    st.addAttributes(def->getAttrs());
                    st.addTypes(lift2DColInner(def->getResult(0).getType(),
                                               S));
                    Operation *nu = rw.create(st);
                    vmap[v] = nu->getResult(0);
                    return nu->getResult(0);
                }
            }
        }
        return Value();
    };

    for (AddPtrSeed seed : seeds) {
        Value load2D = buildAddPtrLoad2D(rw, seed, get2D);
        if (!load2D) return;
        vmap[seed.load.getResult()] = load2D;
    }
    for (triton::LoadOp load : denseLoads) {
        auto baseIt = denseBases.find(load.getOperation());
        auto seedIt = denseSeedIndex.find(load.getOperation());
        if (baseIt == denseBases.end() || seedIt == denseSeedIndex.end())
            return;
        vmap[load.getResult()] = buildDenseLoad2D(rw, load,
                                                  seeds[seedIt->second],
                                                  baseIt->second);
    }

    SmallVector<Operation *> ordered;
    moduleOp.walk([&](Operation *op) {
        if (region.count(op)) ordered.push_back(op);
    });
    for (Operation *op : ordered) {
        rw.setInsertionPoint(op);
        if (auto scan = dyn_cast<triton::ScanOp>(op)) {
            Value in = get2D(scan.getOperand(0));
            if (!in) return;
            auto ns = rw.create<triton::ScanOp>(scan.getLoc(), ValueRange{in},
                                                scan.getAxis() + 1,
                                                scan.getReverse());
            rw.cloneRegionBefore(scan.getCombineOp(), ns.getCombineOp(),
                                 ns.getCombineOp().end());
            vmap[scan->getResult(0)] = ns->getResult(0);
            continue;
        }
        if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
            Value in = get2D(reduce.getOperand(0));
            if (!in) return;
            auto nr = rw.create<triton::ReduceOp>(reduce.getLoc(),
                                                  ValueRange{in},
                                                  reduce.getAxis() + 1);
            rw.cloneRegionBefore(reduce.getCombineOp(), nr.getCombineOp(),
                                 nr.getCombineOp().end());
            vmap[reduce->getResult(0)] = nr->getResult(0);
            continue;
        }
        if (isa<triton::SplatOp>(op)) continue;
        SmallVector<Value> operands;
        for (Value operand : op->getOperands()) {
            Value n = get2D(operand);
            if (!n) return;
            operands.push_back(n);
        }
        OperationState st(op->getLoc(), op->getName());
        st.addOperands(operands);
        st.addAttributes(op->getAttrs());
        for (Value result : op->getResults())
            st.addTypes(lift2DColInner(result.getType(), S));
        Operation *nu = rw.create(st);
        for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
            vmap[oldR] = newR;
    }

    for (const AddPtrStore &store : coalescedStores) {
        triton::StoreOp st = store.store;
        Value val = get2D(st.getValue());
        if (!val) return;
        if (!buildAddPtrStore2D(rw, store, val, S, BT))
            return;
    }

    rw.replaceAllUsesWith(seeds.front().access.batch,
                          seeds.front().access.outputRow);

    for (triton::StoreOp st : sinks) rw.eraseOp(st);
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
        rw.eraseOp(*it);
    for (AddPtrSeed seed : seeds)
        rw.eraseOp(seed.load);

    auto i32t = IntegerType::get(moduleOp.getContext(), 32);
    moduleOp->setAttr("hacc.coalesce_factor", IntegerAttr::get(i32t, S));
    moduleOp->setAttr("hacc.coalesce_axis",
                      IntegerAttr::get(i32t, coalesceAxis));
}

void rewriteStridedAxisCoalesce(ModuleOp moduleOp) {
    rewriteBlockPtrStridedAxisCoalesce(moduleOp);
    if (moduleOp->hasAttr("hacc.coalesce_factor"))
        return;
    rewriteAddPtrStridedAxisCoalesce(moduleOp);
}

}  // namespace StridedAxisCoalescing
