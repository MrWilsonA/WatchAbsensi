```markdown
# Master Project Specification: WatchAbsensi

## 1. Project Overview & Context

### 1.1 Problem Statement
Sistem absensi berbasis pengenalan wajah konvensional umumnya mengandalkan pemindaian gambar statis (upload foto) atau algoritma pencocokan sederhana yang rentan terhadap penipuan identitas (*photo/screen spoofing*). Selain itu, sistem lama sering kali tidak mampu menangani komputasi video stream *real-time* dengan latensi rendah sekaligus mengelola data analitik operasional kehadiran karyawan secara terpadu.

### 1.2 Solution Concept
**WatchAbsensi** adalah platform presensi biometrik cerdas berbasis *live webcam stream* yang memadukan backend performa tinggi **.NET Core**, inferensi model **Computer Vision & Deep Learning**, serta engine **Data Analytics** untuk mengolah metrik kedisiplinan dan anomali kehadiran.

Sistem bekerja secara *real-time*:
* Feed video dari webcam ditangkap melalui browser/kios client.
* Modul AI melakukan verifikasi keaslian wajah (*anti-spoofing / liveness detection*) sebelum mengekstrak *vector embeddings* wajah via deep neural network.
* Hasil pencocokan biometrik dikirim ke backend .NET via event-driven messaging (SignalR / gRPC).
* Seluruh log kehadiran, durasi kerja, dan anomali keterlambatan diagregasi dalam pipeline data analytics terstruktur.

---

## 2. Core Modules & Architecture

### 2.1 Live Video & Anti-Spoofing Pipeline (Computer Vision & Deep Learning)
* **Face Detection & Alignment:** Deteksi lokasi wajah secara real-time pada frame video menggunakan model berbasis CNN ringan (YOLOv8-Face atau MTCNN) dengan OpenCV.
* **Passive/Active Liveness Detection:** Model deep learning memverifikasi apakah target adalah manusia hidup (analisis tekstur kulit, micro-blinking, atau deteksi pantulan layar HP/foto cetak) untuk mencegah kecurangan absensi.
* **Feature Extraction & Vector Matching:** Mengekstrak 512-dimensi *facial embedding* menggunakan arsitektur deep learning modern (ArcFace / MobileFaceNet via ONNX Runtime).
* **Vector Search:** Pencocokan identitas dilakukan via cosine similarity terhadap database vektor lokal (*high accuracy, sub-second inference*).

### 2.2 Core Business & Real-Time Orchestration (.NET Backend)
* **ASP.NET Core Web API:** Dibangun dengan *Clean Architecture* untuk mengelola data master karyawan, shift kerja, kebijakan toleransi keterlambatan, dan permission.
* **SignalR Hub:** Menyediakan koneksi WebSocket dua arah ke antarmuka absensi (kios layar) untuk memberikan feedback audio-visual instan ("Wajah Terverifikasi", "Spoof Terdeteksi", "Sudah Absen").
* **Anti-Double Tap & Debounce Worker:** Memastikan satu individu tidak memicu multi-transaksi absensi dalam rentang waktu yang sama.

### 2.3 Workforce Analytics Engine (Data Analytics)
* **Punctuality & Shift Aggregation:** Mengolah log time-series *check-in* dan *check-out* menjadi metrik keterlambatan, *dwell time*, dan jam kerja efektif per divisi.
* **Anomaly & Fraud Detection Analytics:** Memetakan pola upaya *spoofing* atau kegagalan verifikasi per kamera/lokasi kios.
* **Reporting & Trends Dashboard:** Menyajikan visualisasi tren kehadiran mingguan/bulanan, heatmap jam sibuk absensi, dan performa retensi jam kerja tim.

---

## 3. Technology Stack

| Layer | Komponen / Library | Fungsi Utama |
| :--- | :--- | :--- |
| **Enterprise Backend & API** | .NET 8 / ASP.NET Core Web API | Core business logic, otentikasi JWT, API gateway, dan data persistence |
| **Real-Time Communication** | ASP.NET Core SignalR | WebSocket streaming untuk status absensi instan di layar kios |
| **AI Inference Service** | Python 3.11 (FastAPI) + ONNX Runtime | Eksekusi model deep learning berkecepatan tinggi dengan akselerasi CPU/CUDA |
| **Computer Vision Engine** | OpenCV, Mediapipe, PyTorch | Frame decoding, face cropping, transformasi geometri, dan normalisasi citra |
| **Face Recognition & Liveness** | ArcFace (ONNX), MiniFASNet | Ekstraksi embedding wajah dan klasifikasi anti-spoofing |
| **Database & Vector Store** | PostgreSQL 16 + `pgvector` | Menyimpan data transaksional relasional sekaligus vektor embedding wajah |
| **Data Analytics Pipeline** | Python (Pandas, NumPy) / DuckDB | Agregasi data time-series kehadiran dan perhitungan analitik kedisiplinan |
| **Frontend Kios / Dashboard** | Blazor WebAssembly / React (TailwindCSS) | Antarmuka live webcam feed dan dashboard monitoring admin |
| **Infrastructure** | Docker & Docker Compose | Kontainerisasi multi-service |

---

## 4. Directory & Service Structure

```text
watchabsensi/
├── docker/
│   ├── dotnet-api/
│   │   └── Dockerfile
│   ├── cv-engine/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── nginx/
│       └── default.conf
├── src/
│   ├── backend-dotnet/           # ASP.NET Core Solution
│   │   ├── WatchAbsensi.Domain/
│   │   ├── WatchAbsensi.Application/
│   │   ├── WatchAbsensi.Infrastructure/
│   │   ├── WatchAbsensi.WebApi/
│   │   └── WatchAbsensi.sln
│   ├── cv-engine-python/         # Computer Vision & Inference Service
│   │   ├── app/
│   │   │   ├── detector/         # YOLO-Face / OpenCV pipeline
│   │   │   ├── liveness/         # Anti-spoofing classifier
│   │   │   ├── recognizer/       # ArcFace ONNX inference
│   │   │   ├── analytics/        # Time-series attendance analytics
│   │   │   └── main.py
│   │   ├── models/               # Pretrained weights (.onnx)
│   │   └── requirements.txt
│   └── frontend-client/          # Web Client untuk Live Webcam Feed
├── docker-compose.yml
├── .env.example
└── README.md

```

---

## 5. Docker Infrastructure Blueprint

### 5.1 `docker-compose.yml`

```yaml
services:
  # Database Relasional + Vector Storage
  postgres-db:
    image: pgvector/pgvector:pg16
    container_name: watchabsensi_db
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${DB_NAME:-watchabsensi}
      POSTGRES_USER: ${DB_USER:-postgres}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secret}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - watchabsensi-network

  # In-Memory Cache & Message Broker
  redis:
    image: redis:7-alpine
    container_name: watchabsensi_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
    networks:
      - watchabsensi-network

  # Computer Vision & Deep Learning Engine (Python + ONNX Runtime)
  cv-engine:
    build:
      context: ./src/cv-engine-python
      dockerfile: ../../docker/cv-engine/Dockerfile
    container_name: watchabsensi_cv
    restart: unless-stopped
    environment:
      - PORT=8002
      - DB_CONNECTION_STRING=postgresql://${DB_USER:-postgres}:${DB_PASSWORD:-secret}@postgres-db:5432/${DB_NAME:-watchabsensi}
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./src/cv-engine-python:/app
      - ./models:/app/models
    ports:
      - "8002:8002"
    depends_on:
      - postgres-db
      - redis
    networks:
      - watchabsensi-network

  # Core Backend (.NET 8 Web API)
  dotnet-api:
    build:
      context: ./src/backend-dotnet
      dockerfile: ../../docker/dotnet-api/Dockerfile
    container_name: watchabsensi_api
    restart: unless-stopped
    environment:
      - ASPNETCORE_ENVIRONMENT=Development
      - ConnectionStrings__DefaultConnection=Host=postgres-db;Database=watchabsensi;Username=postgres;Password=secret
      - ConnectionStrings__Redis=redis:6379
      - Services__CvEngineUrl=http://cv-engine:8002
    ports:
      - "5000:8080"
    depends_on:
      - postgres-db
      - redis
      - cv-engine
    networks:
      - watchabsensi-network

networks:
  watchabsensi-network:
    driver: bridge

volumes:
  pgdata:
  redisdata:

```

---

### Dockerfile .NET 8 (`docker/dotnet-api/Dockerfile`)

```dockerfile
# Build Stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /source

COPY *.sln .
COPY WatchAbsensi.Domain/*.csproj ./WatchAbsensi.Domain/
COPY WatchAbsensi.Application/*.csproj ./WatchAbsensi.Application/
COPY WatchAbsensi.Infrastructure/*.csproj ./WatchAbsensi.Infrastructure/
COPY WatchAbsensi.WebApi/*.csproj ./WatchAbsensi.WebApi/

RUN dotnet restore

COPY . .
WORKDIR /source/WatchAbsensi.WebApi
RUN dotnet publish -c Release -o /app/publish --no-restore

# Runtime Stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
COPY --from=build /app/publish .

ENV ASPNETCORE_HTTP_PORTS=8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "WatchAbsensi.WebApi.dll"]

```

---

### Dockerfile CV Engine (`docker/cv-engine/Dockerfile`)

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

```

---

## 6. End-to-End Workflow

1. **Streaming Frame:** Klien webcam (browser/kios) membuka stream video dan mengirimkan frame berurutan ke gateway via WebSocket.
2. **Inference Pipeline (CV & Deep Learning):**
* Frame didekode oleh modul OpenCV.
* Model anti-spoofing mengecek validitas fisik wajah (mencegah manipulasi foto/layar ponsel).
* ArcFace mengekstrak 512-dimensi *embedding* dari wajah yang terdeteksi.


3. **Pencocokan Identitas:** Vektor wajah dicocokkan ke PostgreSQL menggunakan ekstensi `pgvector` dengan operator kemiripan kosinus (`<=>`).
4. **Verifikasi & Notifikasi (.NET):**
* .NET Core API memverifikasi status jadwal kerja dan batas waktu presensi karyawan.
* Hub SignalR memancarkan event langsung ke layar absensi ("Presensi Berhasil: [Nama Karyawan]").


5. **Data Analytics Pipeline:**
* Setiap catatan presensi disimpan bersama atribut waktu, durasi keterlambatan, dan confidence score.
* Worker analitik mengagregasi data untuk metrik kepatuhan jam kerja, pola anomali harian, serta tren absensi departemen.