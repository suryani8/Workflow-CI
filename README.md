# Workflow-CI

## Deskripsi
Repository ini berisi workflow CI/CD untuk otomatisasi training model machine learning Heart Disease menggunakan MLflow Project dan Docker.

## Struktur Repository
Workflow-CI/
├── .github/workflows/    # GitHub Actions (CI berjalan di sini)
│   └── ci.yml
├── .workflow/            # Workflow (ketentuan submission)
│   └── ci.yml
└── MLProject/            # MLflow Project
├── modelling.py
├── conda.yaml
├── MLProject
├── heart_preprocessing_train.csv
└── heart_preprocessing_test.csv

## Workflow CI
Workflow otomatis berjalan ketika ada push ke branch `main` pada folder `MLProject/`.

### Tahapan:
1. Checkout Repository
2. Setup Python 3.12
3. Check Environment
4. Install Dependencies
5. Run MLflow Project
6. Get Latest MLflow Run ID
7. Install Python Dependencies
8. Upload Artifacts to GitHub
9. Login to Docker Hub
10. Build Docker Model
11. Tag Docker Image
12. Push Docker Image

## MLflow Tracking
Hasil training tersimpan di DagsHub:
👉 https://dagshub.com/suryani8/Eksperimen_SML_Suryani_apc367d6x0436

## Docker Image
Image tersedia di Docker Hub:
👉 https://hub.docker.com/r/suryanii/heart-disease-rf
