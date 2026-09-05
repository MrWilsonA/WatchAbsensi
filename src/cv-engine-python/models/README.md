# Model Weights: ArcFace & Anti-Spoofing

Direktori ini digunakan untuk menyimpan file model machine learning dalam format ONNX Runtime (`.onnx`).

### File Model yang Didukung:
1. `arcface.onnx` / `w600k_r50.onnx`:
   - Arsitektur: ResNet-50 / MobileFaceNet ArcFace
   - Input: Citra wajah terpotong dan ter-align ukuran `(1, 3, 112, 112)`, range `[-1.0, 1.0]`
   - Output: Vektor embedding 512-dimensi ternormalisasi L2
   - Fungsi: Pencocokan biometrik berkecepatan tinggi via Cosine Similarity (`dot product >= 0.75`).

2. `minifasnet.onnx`:
   - Arsitektur: MiniFASNetV2 / Silent-Face-Anti-Spoofing
   - Input: Crop wajah multi-skala
   - Output: Klasifikasi biner (Real Human Face vs Spoof/Screen/Print).

> **Catatan Pengembangan Offline**: Jika file model `.onnx` belum ditempatkan pada folder ini, sistem CV Engine secara otomatis menggunakan mode **Deterministic Perceptual Projection**. Mode ini mengekstrak vektor 512-dimensi stabil berbasis Discrete Cosine Transform (DCT) dan image hashing sehingga seluruh fitur pencocokan, registrasi karyawan, dan absensi dapat diuji langsung tanpa kendala dependensi download besar.
