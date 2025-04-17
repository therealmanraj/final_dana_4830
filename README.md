# HVAC Energy Consumption Model — CI/CD & Real‑Time Inference Pipeline

## Overview

This repository implements an end-to-end pipeline for training, deploying, and serving an HVAC energy‑consumption forecasting model using AWS SageMaker, CodeBuild, CodePipeline, S3, and Lambda:

1. **Continuous Training**

   - New data uploaded to `s3://<bucket>/training-data/` triggers CodePipeline.
   - CodeBuild executes `buildspec.yml`, launching a SageMaker training job, creating a new model, updating the existing SageMaker endpoint in-place.

2. **Real‑Time Inference**
   - New CSVs placed in `s3://<bucket>/incoming/` fire an S3 event -> Lambda.
   - The Lambda function invokes the SageMaker endpoint, saving predictions to `s3://<bucket>/predictions/`.

![Architecture Diagram](/images/HVAC.jpg)

---

## S3 Bucket Layout

```
s3://<your-bucket>/
├── training‑data/      # Raw data for periodic retraining
├── model‑artifacts/    # SageMaker training output
├── incoming/           # New files for real‑time inference
└── predictions/        # Model output from Lambda
```

---

## CI/CD Pipeline

- **GitHub Integration**:  
  Configured via AWS Connector for GitHub App: automatically rebuild on code push.

- **CodeBuild (`buildspec.yml`)**:

  1. Reads environment variables (S3 URIs, IAM role, endpoint name)
  2. Starts a SageMaker training job
  3. Waits for completion
  4. Creates a new SageMaker model & endpoint config
  5. Updates the existing endpoint in-place

- **CodePipeline**:
  - **Source**: GitHub (main branch)
  - **Build**: CodeBuild project `hvac-model-build`
  - **Deploy**: Updates SageMaker endpoint with no downtime

---

## Lambda‑Based Inference

- **Trigger**: S3 `ObjectCreated:Put` events on `incoming/`
- **Function**:

  1. Downloads the CSV
  2. Invokes SageMaker endpoint via `runtime.invoke_endpoint`
  3. Saves returned CSV to `predictions/`

- **IAM Permissions**:
  - Lambda role must allow `s3:GetObject` / `s3:PutObject` and `sagemaker:InvokeEndpoint`

---

## Prerequisites & IAM Roles

- **SageMakerExecutionRole** — allows SageMaker to read training data, write artifacts, pull Docker images from ECR.
- **LambdaTriggerRoleS3** — allows Lambda to read from `incoming/`, invoke SageMaker, write to `predictions/`.
- **CodeBuild‑HVAC‑BuildRole** — allows creating SageMaker jobs, updating endpoints, reading from ECR, S3.

---

## Setup & Deployment

1. **Clone** this repo.
2. **Create** an S3 bucket with the above prefixes.
3. **Configure** IAM roles and environment variables:
   - `S3_TRAINING_URI`, `MODEL_ARTIFACTS_S3`, `ENDPOINT_NAME`, `IAM_ROLE_ARN`
4. **Connect** your GitHub repo via AWS Connector in CodeBuild and CodePipeline.
5. **Create**:
   - CodeBuild project (`hvac-model-build`)
   - CodePipeline (`hvac-model-pipeline`) with source → build stages
6. **Deploy** Lambda function with S3 trigger on `incoming/`.

---

## File Structure

```
.
├── HVAC_Model/
│   ├── buildspec.yml       # CodeBuild spec
│   ├── train.py            # Training script for local reference
│   ├── inference.py        # SageMaker inference handler
│   └── requirements.txt    # Python dependencies
├── lambda_function.py      # Lambda handler for real-time inference
├── build_train_deploy.ipynb # Jupyter notebook for training & deployment
└── README.md               # This documentation
```

---

## How It Works

1. New commit → CodePipeline → CodeBuild → SageMaker retraining → in‑place endpoint update
2. Periodic or ad-hoc retraining via pushing new data to `training-data/`
3. Real‑time scoring by uploading to `incoming/` → Lambda → SageMaker → `predictions/`

---

For questions or further customization, please open an issue or contact the author.
