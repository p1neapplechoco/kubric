# Kubric Demo: Hướng dẫn setup và chạy (VI)

Tài liệu này dành cho mục tiêu chạy demo nhanh của Kubric (hello world + simulation video MP4).

## 0) Chạy nhanh nhất (one-click)

Repo đã có script:

```bash
cd /home/pineapple/Desktop/projects/kubric
./run_demo.sh hello   # demo ảnh tĩnh
./run_demo.sh sim     # demo simulation + output/simulator.mp4
./run_demo.sh all     # chạy cả 2 demo
```

Xem trợ giúp:

```bash
./run_demo.sh --help
```

## 1) Chuẩn bị môi trường

### Cách đang dùng ổn định nhất (khuyến nghị): Docker

Yêu cầu:
- Docker đã cài và chạy được (`docker --version`)

Vào thư mục project:

```bash
cd /home/pineapple/Desktop/projects/kubric
```

### Môi trường local riêng tên `thesis`

Repo này đã tạo sẵn local venv `thesis` để tách môi trường:

```bash
cd /home/pineapple/Desktop/projects/kubric
source thesis/bin/activate
```

Lưu ý: trong máy hiện tại chưa có `conda`, nên đang dùng `venv` làm môi trường dedicated.

## 2) Chạy demo ảnh tĩnh (hello world)

```bash
cd /home/pineapple/Desktop/projects/kubric
docker run --rm --interactive \
  --user $(id -u):$(id -g) \
  --volume /home/pineapple/Desktop/projects/kubric:/workspace \
  kubricdockerhub/kubruntudev \
  python3 examples/helloworld.py
```

Output chính:
- `output/helloworld.png`
- `output/helloworld_depth.png`
- `output/helloworld_segmentation.png`
- `output/helloworld.blend`

## 3) Chạy demo simulation + render video MP4

`examples/simulator.py` đã được chỉnh để xuất thêm `output/simulator.mp4`.

Do image `kubruntudev` mặc định chưa có backend MP4, script `run_demo.sh sim` sẽ tự cài runtime trong container.

Nếu muốn chạy thủ công (không qua script), dùng lệnh:

```bash
cd /home/pineapple/Desktop/projects/kubric
docker run --rm --interactive \
  --volume /home/pineapple/Desktop/projects/kubric:/workspace \
  kubricdockerhub/kubruntudev \
  sh -lc 'python3 -m pip install --quiet --disable-pip-version-check imageio-ffmpeg && python3 examples/simulator.py'
```

Output chính:
- `output/simulator.mp4`
- `output/simulator.blend`
- bộ frame/layer PNG trong thư mục `output/`

## 4) Khi chạy demo thì thực tế sẽ làm gì?

### Với `examples/helloworld.py`
1. Tạo scene đơn giản (sàn, 1 quả cầu, đèn, camera)
2. Blender render 1 frame
3. Ghi PNG màu, depth, segmentation và file `.blend`

### Với `examples/simulator.py`
1. Tạo scene vật lý (PyBullet) với nhiều quả cầu ngẫu nhiên
2. Chạy mô phỏng để sinh quỹ đạo/keyframe
3. Blender render toàn bộ dải frame
4. Ghi các layer ảnh ra `output/`
5. Ghép chuỗi RGBA thành `output/simulator.mp4`

## 5) Kiểm tra nhanh sau khi chạy

```bash
cd /home/pineapple/Desktop/projects/kubric
ls -lh output/simulator.mp4 output/simulator.blend
```

## 6) Troubleshooting nhanh

- Lỗi Docker socket / permission:
  - Đảm bảo user có quyền Docker hoặc chạy Docker daemon đúng cách.
- Không thấy `simulator.mp4`:
  - Chạy đúng lệnh ở mục 3 (có bước cài `imageio-ffmpeg` trong container).
- Có warning kiểu `Not freed memory blocks` cuối log Blender:
  - Warning này có thể xuất hiện nhưng thường không chặn output nếu file đã được ghi thành công.
