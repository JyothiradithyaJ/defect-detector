# EV Charging Station Assembled Panel Inspection: Wire Disconnection & Two-Stage Mapping

## 1. How "Wire Disconnected, Then Deeper Checks" Maps to Stage A + Stage B

In an EV charging station assembled panel (`APD` — Assembled Panel Dataset for EV charging stations), inspection follows the two-stage hierarchy defined in `README_v3.md`:

1. **Stage 0 (`v3/core/qc.py`) — Capture Quality Gate**:
   - Rejects blurry, under/over-exposed, or poorly framed photos with verdict `RECAPTURE` (never `REJECT`) before any component or surface check runs.
2. **Stage 1 (`v3/core/align.py`) — Template Registration**:
   - Registers the query photo onto the golden normal panel template (`ORB + RANSAC` homography followed by `ECC` refinement inside `stable_roi`). If alignment residual exceeds `max_align_residual`, returns `misaligned` rather than running positional ROI checks on a shifted coordinate frame.
3. **Stage A (`v3/core/roi.py`) — Logical & Assembly Verification ("Wire Disconnected / Wrong Slot")**:
   - **`count` ROI (`sensor_right_wires`)**: Counts connected terminal/wire blobs inside the terminal block polygon and compares against `expected_count`. A disconnected or missing wire immediately raises a `count` mismatch (`expected_count` vs `blob_count`).
   - **`presence` ROI (`red_cap_left`)**: Checks edge-density and intensity z-scores against reference normals to verify whether a required cap, lug, or connector is present.
   - **`order` ROI (`ev_port_left_cables`)**: Splits multi-wire harnesses into ordered slots along `axis` (`x` or `y`) and verifies the colour/appearance sequence against reference normals (`wrong_cable_order`).
   - **`position` ROI**: Checks the intensity centroid shift (in pixels) of a mounted component inside its tolerance polygon.
4. **Stage B (`v3/core/defect/knn.py` + `v3/core/fusion.py`) — Deeper Surface & Unforeseen Defect Check**:
   - Even when all wires and caps pass Stage A (`mismatches = []`), or when `early_exit: false` is set (default in `v3/config/default.yaml`), Stage B runs DINOv2 (`dinov2_vitl14_reg` @ 560 px, `w_d = 0.70`) fused with CLIP (`ViT-L-14-336` @ 336 px, `w_c = 0.30`) across the full panel and inside `surface` ROIs (`pcb_left`).
   - This catches burn marks, PCB solder/trace damage (`pcb_surface_damage`), insulation nicks, or enclosure scratches (`panel_scratch_dent`) that do not alter wire counts or slot ordering.

---

## 2. What Is NOT Yet Validated on Real EV Charging Panel Data

1. **APD (`D:\data\apd`) Not Yet Run**:
   - The Assembled Panel Dataset (`D:\data\apd`) is not yet placed on disk. No real EV charging station panel images have been scored by Stage A or Stage B yet.
2. **Stage A Precision/Recall (`1.0000`) Was Measured on Synthetic Perturbations Only**:
   - The `1.0000` precision/recall reported in Phase 4 (`v3/eval/step4_stage_a_eval.py`) was computed on a small synthetic 256×256 assembly board generator (12 reference normals, 40 validation normals, and 40 synthetic anomalies: 10 per ROI kind) where the ROI polygons and perturbation locations were constructed in the same script.
   - Real logical-anomaly validation on `MVTec LOCO` (`D:\data\mvtec_loco`) and `APD` (`D:\data\apd`) remains pending until those datasets are placed on disk.
3. **APD ROI Polygons Still Require Hand-Drawing from Documentation / Golden Template**:
   - `v3/config/recipes/apd_panel.yaml` currently has `status: needs-human-roi-drawing` and empty polygon lists (`polygon: []`, `expected_count: null`).
   - Before running `step6_apd_protocol.py` or `demo_single_image.py` on real APD images, a human reviewer must draw the ROI polygons on the golden normal template (`python -m v3.eval.draw_rois`) using engineering documentation and normal reference images only (never test/defective images), and set `status: human-reviewed-and-frozen`.

