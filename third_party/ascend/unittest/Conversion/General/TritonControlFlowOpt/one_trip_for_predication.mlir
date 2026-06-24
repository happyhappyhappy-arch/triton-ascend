// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s

module {
  tt.func public @predicate_one_trip_loads(%base: !tt.ptr<f32>, %dense: !tt.ptr<f32>, %len: index, %offsets: tensor<4xi32>, %col_mask: tensor<4xi1>) -> tensor<4xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %trip = arith.minsi %len, %c1 : index
    %res:2 = scf.for %iv = %c0 to %trip step %c1 iter_args(%acc = %cst, %ptr = %base) -> (tensor<4xf32>, !tt.ptr<f32>) {
      %iv_i64 = arith.index_cast %iv : index to i64
      %scalar_ptr = tt.addptr %dense, %iv_i64 : !tt.ptr<f32>, i64
      %scalar = tt.load %scalar_ptr : !tt.ptr<f32>
      %splat_ptr = tt.splat %ptr : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
      %vec_ptr = tt.addptr %splat_ptr, %offsets : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %vec = tt.load %vec_ptr, %col_mask, %cst : tensor<4x!tt.ptr<f32>>
      %splat_scalar = tt.splat %scalar : f32 -> tensor<4xf32>
      %prod = arith.mulf %splat_scalar, %vec : tensor<4xf32>
      %sum = arith.addf %acc, %prod : tensor<4xf32>
      %next = tt.addptr %ptr, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %sum, %next : tensor<4xf32>, !tt.ptr<f32>
    }
    tt.return %res#0 : tensor<4xf32>
  }
}

// CHECK-LABEL: tt.func public @predicate_one_trip_loads
// CHECK-NOT:   scf.for
// CHECK:       %[[COND:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK:       %[[SCALAR_ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:       tt.load %{{.*}}, %[[COND]], %[[SCALAR_ZERO]] : !tt.ptr<f32>
// CHECK:       %[[PRED_MASK:.*]] = tt.splat %[[COND]] : i1 -> tensor<4xi1>
// CHECK:       %[[COMBINED_MASK:.*]] = arith.andi %{{.*}}, %[[PRED_MASK]] : tensor<4xi1>
// CHECK:       tt.load %{{.*}}, %[[COMBINED_MASK]], %{{.*}} : tensor<4x!tt.ptr<f32>>
// CHECK:       arith.select %[[COND]], %{{.*}}, %{{.*}} : tensor<4xf32>
// CHECK:       tt.return

// -----

module {
  tt.func public @predicate_one_trip_store(%ptrs: tensor<4x!tt.ptr<f32>>, %value: tensor<4xf32>, %len: index, %mask: tensor<4xi1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %trip = arith.minsi %len, %c1 : index
    scf.for %iv = %c0 to %trip step %c1 {
      tt.store %ptrs, %value, %mask : tensor<4x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @predicate_one_trip_store
// CHECK-NOT:   scf.for
// CHECK:       %[[COND:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}} : index
// CHECK:       %[[PRED_MASK:.*]] = tt.splat %[[COND]] : i1 -> tensor<4xi1>
// CHECK:       %[[COMBINED_MASK:.*]] = arith.andi %{{.*}}, %[[PRED_MASK]] : tensor<4xi1>
// CHECK:       tt.store %{{.*}}, %{{.*}}, %[[COMBINED_MASK]] : tensor<4x!tt.ptr<f32>>
// CHECK:       tt.return
