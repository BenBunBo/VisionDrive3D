# Chẩn đoán finetune YOLO26n-seg — 2026-10-15

Nguyên nhân nổi bật là lỗi tạo nhãn instance segmentation và sự không nhất quán trong đánh giá. Dữ liệu đang gán nhầm mask giữa các xe, đưa một phần nền vào nhãn car, và tách một instance thành nhiều đối tượng. Vì cả train, validation và benchmark đều dùng nhãn phát sinh từ quy trình này, điểm validation cao chưa chứng minh chất lượng segmentation thực tế.

Đã kiểm tra 1.000 ảnh và nhãn tương ứng; đọc cấu hình, cache nhãn, checkpoint và lịch sử 50 epoch; chạy lại hai checkpoint baseline/finetuned trên 100 ảnh test bằng NVIDIA A40. Kết quả benchmark cũ được tái lập chính xác. Các phép đánh giá thay đổi ground truth dưới đây chỉ nhằm phân lập lỗi, chưa phải benchmark trên ground truth đã làm sạch.

## 1. Nền bị render bằng màu instance của xe số 7

Chuỗi code gây lỗi:

- [MaskRenderer.render](../../src/render/renderers.py#L74) đặt uniform `instance_color` lần lượt theo từng entity. Sau vòng lặp, uniform giữ màu xe cuối cùng, ID 7.
- [Export mask](../../src/render/renderers.py#L286) tiếp tục vẽ scene overlay bằng cùng shader.
- [RoadMeshDrawable.draw](../../src/scene/api_scene_overlay.py#L440) chỉ cập nhật ma trận và texture trong nhánh `is_rgb=False`, không đặt lại `instance_color`.
- [SceneOverlay.render](../../src/scene/scene_overlay.py#L228) cũng vẽ building entities mà không upload `ent.instance_color`.

Màu ID 7 là `(215, 83, 132)` theo mảng đọc bằng OpenCV. Trong `scene_00025`, màu này chiếm **668.755 pixel**, bao gồm mặt đường và cột đèn. ID 7 của scene này có bbox `[0, 0, 0, 0]` trong metadata. Đây là bằng chứng trực tiếp màu xe xuất hiện trên nền.

**968/1.000 ảnh** có hơn 10.000 pixel màu ID 7 nằm ngoài hợp của mọi bbox xe trong metadata: 773/800 train, 98/100 val, 97/100 test. Đường đi của code phù hợp với lỗi quan sát trên mask đã lưu; chưa chạy lại renderer sau sửa lỗi.

Hệ quả: không thể giải mã toàn ảnh bằng palette ID 7 rồi coi đó là mask xe sạch. Crop trong bbox chỉ giới hạn vùng lỗi, không loại bỏ lỗi.

## 2. Chọn màu đông pixel nhất trong bbox không xác định đúng instance

[build_car_polygon_labels](../../src/ai_demo/finetune_yolo26_qualitative.py#L67) chọn màu có nhiều pixel nhất trong bbox. [Benchmark](../../src/ai_demo/benchmark_yolo26_finetune.py#L68) dùng cùng nguyên tắc.

Đối chiếu thứ tự car trong metadata với thứ tự pool tạo màu trong renderer; đã kiểm tra bbox metadata tái tạo đúng tất cả nhãn YOLO của 1.000 ảnh sau làm tròn 6 chữ số:

| Split | Box car | Box tìm được mask | Chọn sai ID so với xe đích | Tỷ lệ trên mask tìm được |
|---|---:|---:|---:|---:|
| Train | 2.070 | 2.002 | **527** | **26,32%** |
| Val | 283 | 279 | 79 | 28,32% |
| Test | 232 | 226 | 64 | 28,32% |

Trong 527 trường hợp sai ở train, 509 vẫn có pixel của xe đích trong crop. Có 56 trường hợp chọn nhầm sang ID 7, vốn bị lẫn với nền. Việc ID 7 được chọn không tự nó phân biệt được nền với xe số 7 thật.

Ví dụ:

- `scene_00025`: bbox xe số 5 chứa 1.352 pixel của chính xe đó, nhưng chứa 2.119 pixel xe số 2 che phía trước; bộ chuyển nhãn chọn xe số 2.
- `scene_00030`: bbox xe số 1 chứa 4.900 pixel của xe, nhưng 10.528 pixel màu ID 7 ở phần nền; nhãn trở thành vùng nền quanh xe.

Ngoài ra, train có **156 cặp mask** được gán cùng màu và có IoU > 0,5. Việc học trên nhãn trùng/lệch đối tượng có thể làm mô hình bỏ xe bị che khuất và học mask sai. Chưa có thí nghiệm retrain trên dữ liệu sạch để định lượng riêng mức tổn thất do từng loại nhiễu.

![Hai ví dụ chọn nhầm mask](../../.mesh_cache/yolo26_investigation/label_failure_examples.png)

Ảnh minh họa crop theo bbox metadata nguyên pixel; thống kê audit mô phỏng chính xác crop sau chuẩn hóa/làm tròn YOLO nên ví dụ đầu có chênh 5 pixel ở màu được chọn.

## 3. Một mask bị chuyển thành nhiều instance

[Vòng lặp contour](../../src/ai_demo/finetune_yolo26_qualitative.py#L73) dùng `RETR_EXTERNAL` rồi ghi **mỗi contour thành một dòng nhãn**. Theo [định dạng Ultralytics](https://docs.ultralytics.com/datasets/segment/), mỗi dòng là một object instance.

Ở train, 188/2.002 mask có nhiều contour đạt ngưỡng; 3 mask bị loại hoàn toàn bởi ngưỡng diện tích. Kết quả là **2.193 polygon** từ 2.002 mask. Ở test, 226 mask trở thành 246 polygon. Trong benchmark, các vùng rời vẫn được giữ chung trong một RLE mask cho mỗi bbox.

Như vậy train và benchmark khác nhau ngay ở định nghĩa một instance. Việc này có thể do che khuất và cũng có thể do mask nền bị chọn sai. `RETR_EXTERNAL` còn bỏ cấu trúc lỗ; audit này chưa định lượng riêng ảnh hưởng của việc lấp lỗ.

## 4. Box training và box benchmark khác quy ước

[project_entity_bbox_2d](../../src/annotation/annotations.py#L132) chiếu 8 đỉnh OBB 3D để tạo nhãn detection. Trong khi đó Ultralytics suy box training segmentation từ polygon (`segments2boxes` trong bản cài 8.4.173).

Giữ nguyên dự đoán, chỉ đổi ground truth bbox:

| Ground truth box | Baseline AP50–95 | Finetuned AP50–95 |
|---|---:|---:|
| Box chiếu 3D đang dùng ở benchmark | 32,30 | 32,15 |
| Box ôm sát mask benchmark hiện tại | 43,61 | **72,19** |
| Box từ polygon training của tập test | 46,17 | **82,17** |

Điều này giải thích phần lớn hiện tượng box AP không tăng: mô hình đã học quy ước khác với thước đo. Các giá trị 72,19/82,17 vẫn dùng nhãn có lỗi nêu trên, **không phải chất lượng được xác nhận trên ground truth sạch**.

## 5. Benchmark chưa bỏ letterbox padding trước khi resize mask

[benchmark_yolo26_finetune.py](../../src/ai_demo/benchmark_yolo26_finetune.py#L137) resize trực tiếp `result.masks.data` về 1600×900. Trong lần chạy thực tế, mask là 640×384; nội dung ảnh là 640×360, có đệm 12 pixel ở mỗi phía trên/dưới.

Đã dùng `ultralytics.utils.ops.scale_masks` để bỏ padding rồi nội suy, giữ nguyên checkpoint, dự đoán và ground truth:

| Cách đưa mask về ảnh gốc | Baseline AP50 | Finetuned AP50 | Baseline AP50–95 | Finetuned AP50–95 |
|---|---:|---:|---:|---:|
| Benchmark hiện tại | 75,91 | 61,99 | 34,48 | 41,36 |
| Bỏ padding đúng | 76,60 | 62,30 | 37,99 | **45,14** |

Lỗi resize làm finetuned mất **3,77 điểm AP50–95**, nhưng sửa lỗi này chưa khắc phục mức giảm AP50. [Ultralytics predict](https://docs.ultralytics.com/modes/predict/) cũng hỗ trợ `retina_masks=True` để trả mask theo kích thước ảnh gốc; đó là phương án khác, chưa được đo trong audit này.

## 6. Chưa có bằng chứng ưu tiên tăng epoch hoặc đổi model

Checkpoint xác nhận: Ultralytics 8.4.173, 50 epoch, batch 8, imgsz 640, AdamW, learning rate đầu 0,002, không freeze; `overlap_mask=True`, `mask_ratio=4`. `args.yaml` ghi `optimizer=auto`, `lr0=0.01`, nhưng **checkpoint lưu optimizer thực tế là AdamW/0,002**.

- Train segmentation loss giảm 3,0456 → 1,1177; validation segmentation loss giảm 2,6068 → 1,3013.
- `best.pt` khớp metrics epoch 40: validation mask AP50–95 = 71,44, box AP50–95 = 85,19.
- Epoch 50 có mask AP50–95 = 69,01. Pipeline dùng đúng `best.pt`, không lấy nhầm `last.pt`.
- Cache xác nhận loader thực sự đọc 800 ảnh/2.193 polygon train và 100 ảnh/307 polygon val.

Các số này cho thấy quá trình tối ưu có tiến triển trên bộ nhãn hiện tại. Validation dùng cùng cách tạo nhãn lỗi nên chưa đủ để đánh giá khả năng tổng quát hóa hoặc kết luận mức overfitting. Chưa có ablation để quy lỗi cho kích thước nano, augmentation, learning rate hay độ phân giải.

## Thứ tự xử lý đề xuất

1. Sửa renderer để mọi drawable đặt rõ semantic/instance ID; dùng nền khác mọi ID xe. Xuất mapping instance ID/màu trong metadata. Render lại mask từ trạng thái scene/camera tương ứng; nếu không tái tạo chính xác được, cần tạo lại bộ dữ liệu đồng bộ RGB/mask/annotation.
2. Tạo nhãn theo ID thật của từng xe, xử lý xe không có pixel nhìn thấy theo một quy tắc chung. Không chọn màu theo đa số trong box. Duy trì một instance cho các vùng rời; lựa chọn biểu diễn polygon phù hợp hoặc hỗ trợ mask nhiều thành phần.
3. Dùng ground truth car instance đã kiểm tra cho cả train/val/test; chấm box segmentation theo vùng nhìn thấy. Nếu cần box chiếu 3D, báo cáo thành bài toán/metric riêng.
4. Sửa khôi phục mask sau letterbox cho cả hai script benchmark segmentation. Tạo lại nhãn/cache và baseline trước khi finetune lại với cấu hình hiện tại.
5. Sau khi pipeline sạch mới so sánh thêm epoch, imgsz lớn hơn, augmentation nhẹ hơn hoặc model lớn hơn.

## Bằng chứng và cách tái lập

- [Audit nhãn](../../.mesh_cache/yolo26_investigation/label_audit.json), [audit identity/nền](../../.mesh_cache/yolo26_investigation/identity_audit.json), [đánh giá đối chứng](../../.mesh_cache/yolo26_investigation/evaluation.json).
- [Script audit](../../.mesh_cache/yolo26_investigation/audit.py), [script identity](../../.mesh_cache/yolo26_investigation/identity_audit.py).
- [Benchmark đã công bố](../assets/data/benchmark_yolo26_finetune.json), [training history](../../.mesh_cache/yolo26_qualitative/runs/segment/yolo26n_seg_50ep_b8/results.csv).

```bash
/home/ml4u/anaconda3/envs/vd3d-ai/bin/python .mesh_cache/yolo26_investigation/audit.py --labels-only
/home/ml4u/anaconda3/envs/vd3d-ai/bin/python .mesh_cache/yolo26_investigation/audit.py --evaluate
/home/ml4u/anaconda3/envs/vd3d-ai/bin/python .mesh_cache/yolo26_investigation/identity_audit.py
```

Các file trong `.mesh_cache` là bằng chứng cục bộ, không được Git theo dõi. `identity_gt.json` là bản trung gian chẩn đoán có ID 7 lẫn nền, không được dùng làm ground truth sạch. Audit không sửa pipeline sản xuất, checkpoint hoặc kết quả benchmark đã công bố; chưa chạy finetune mới.
