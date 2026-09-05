# WatchAbsensi

WatchAbsensi adalah platform workforce attendance profesional dengan kiosk webcam, enrollment wajah multi-sudut, liveness anti-spoofing, pencocokan ArcFace 512-D, SignalR real-time feed, direktori karyawan, dan analytics.

## Jalankan

```bash
docker compose up --build
```

- Dashboard: http://localhost:3001
- .NET API docs: http://localhost:5000/swagger
- CV API docs: http://localhost:8002/docs
- Health: http://localhost:5000/health dan http://localhost:8002/health

Alur enrollment: buka Directory → Enroll This Face → izinkan kamera → ambil wajah depan, kiri, dan kanan. Setiap frame melewati deteksi wajah dan liveness sebelum embedding disimpan. Server membandingkan semua sample dengan template karyawan lain; jika kemiripan ≥ 0.92, proses dihentikan dengan status `already_registered` dan nama pemilik template dikembalikan.

Alur absensi: buka Kiosk → Activate Camera → Auto-Detect Face. Frame dikirim ke CV Engine, lalu embedding dicocokkan ke seluruh sample terdaftar melalui endpoint `.NET /api/v1/attendance/auto-scan`. Hasil berhasil menampilkan nama seperti Willy atau Kenny dan langsung tercatat sesuai mode check-in/check-out.

Jika model ONNX belum tersedia, engine memakai deterministic perceptual fallback untuk development. Untuk produksi, letakkan `arcface.onnx` dan `minifasnet.onnx` di `src/cv-engine-python/models`; data registry in-memory pada contoh ini dapat diganti dengan PostgreSQL + pgvector tanpa mengubah kontrak endpoint.
