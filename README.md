# WatchAbsensi

Platform presensi biometrik real-time dengan endpoint inference/liveness, pencatatan absensi, analytics summary, dan dashboard kiosk web.

## Jalankan

```bash
docker compose up --build
```

- Dashboard: http://localhost:3000
- API docs: http://localhost:8002/docs
- Health: http://localhost:8002/health

API saat ini memakai adapter inference deterministik untuk pengembangan lokal; model ONNX ArcFace/MiniFASNet dapat ditempatkan di `src/cv-engine-python/models` dan dihubungkan pada service inference produksi.
