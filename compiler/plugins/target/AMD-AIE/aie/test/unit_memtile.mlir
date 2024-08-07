
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-DAG:         aie.connect<DMA : 0, North : 0>
// CHECK-DAG:         aie.connect<DMA : 1, North : 1>
// CHECK-DAG:         aie.connect<North : 0, DMA : 0>
// CHECK-DAG:         aie.connect<North : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-DAG:         aie.connect<South : 0, DMA : 0>
// CHECK-DAG:         aie.connect<South : 1, DMA : 1>
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK-DAG:         aie.connect<DMA : 2, North : 0>
// CHECK-DAG:         aie.connect<DMA : 3, North : 1>
// CHECK-DAG:         aie.connect<North : 0, DMA : 2>
// CHECK-DAG:         aie.connect<North : 1, DMA : 3>
// CHECK-DAG:         aie.connect<DMA : 4, North : 2>
// CHECK-DAG:         aie.connect<DMA : 5, North : 3>
// CHECK-DAG:         aie.connect<North : 2, DMA : 4>
// CHECK-DAG:         aie.connect<North : 3, DMA : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK-DAG:         aie.connect<South : 0, DMA : 0>
// CHECK-DAG:         aie.connect<South : 1, DMA : 1>
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK-DAG:         aie.connect<South : 2, North : 0>
// CHECK-DAG:         aie.connect<South : 3, North : 1>
// CHECK-DAG:         aie.connect<North : 0, South : 2>
// CHECK-DAG:         aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK-DAG:         aie.connect<South : 0, DMA : 0>
// CHECK-DAG:         aie.connect<South : 1, DMA : 1>
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:         }

module {
    aie.device(xcve2802) {
        %t04 = aie.tile(0, 4)
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t01 = aie.tile(0, 1)
        aie.flow(%t01, DMA : 0, %t02, DMA : 0)
        aie.flow(%t01, DMA : 1, %t02, DMA : 1)
        aie.flow(%t02, DMA : 0, %t01, DMA : 0)
        aie.flow(%t02, DMA : 1, %t01, DMA : 1)
        aie.flow(%t02, DMA : 2, %t03, DMA : 0)
        aie.flow(%t02, DMA : 3, %t03, DMA : 1)
        aie.flow(%t03, DMA : 0, %t02, DMA : 2)
        aie.flow(%t03, DMA : 1, %t02, DMA : 3)
        aie.flow(%t02, DMA : 4, %t04, DMA : 0)
        aie.flow(%t02, DMA : 5, %t04, DMA : 1)
        aie.flow(%t04, DMA : 0, %t02, DMA : 4)
        aie.flow(%t04, DMA : 1, %t02, DMA : 5)
    }
}
