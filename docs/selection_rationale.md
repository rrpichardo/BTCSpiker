# Selection Rationale: BTCSpiker

## Executive Summary
This project delivers a production-ready, real-time AI service designed to detect volatility spikes in cryptocurrency markets. By integrating high-throughput data ingestion (Kafka), low-latency inference (FastAPI), and robust operational observability (Prometheus, Grafana, Evidently), the system meets enterprise standards for reliability and performance.

## The Problem: Volatility Detection
Cryptocurrency markets are characterized by high-frequency, non-linear volatility. Accurate detection of these spikes is critical for risk management and automated trading strategies. Our goal was to build a system that achieves:
* *Sub-second Latency:* Critical for real-time market reactions.
* *Observability:* Ability to track model drift and system health.
* *Resilience:* Graceful handling of data streams and service recovery.

## Why This Architecture & Model
Our technical stack was chosen to balance predictive performance with operational maintainability.

### 1. The Model: Logistic Regression
We selected a *Logistic Regression pipeline* over more complex deep learning models for the following reasons:
* *Inference Speed:* Logistic Regression provides sub-millisecond inference times, critical for meeting our p95 ≤ 800ms SLO.
* *Explainability:* In financial contexts, understanding why a model predicts a spike is as important as the prediction itself. Linear models offer transparent feature coefficients.
* *Baseline Competence:* It provides a strong, robust baseline that is difficult to overfit, facilitating easier monitoring and drift detection.

### 2. The Infrastructure
* *FastAPI:* Chosen for its asynchronous capabilities, allowing for high-concurrency request handling with minimal overhead.
* *Kafka/Docker:* A containerized, event-driven architecture ensures that the system is decoupled, scalable, and easy to deploy in any environment.
* *Monitoring Stack:* We integrated *Prometheus* and *Grafana* for real-time observability, and *Evidently* for automated drift reporting, ensuring the model's performance remains consistent as market conditions evolve.

## Key Operational Decisions
* *Model Versioning:* We implemented MODEL_VARIANT toggling (ml vs. baseline), allowing for instantaneous rollbacks if performance degrades in production.
* *Observability:* By setting specific SLOs for latency and error rates, the system is designed to trigger alerts before critical failures occur.
* *CI/CD:* The automated CI pipeline (Black/Ruff/Integration Tests) ensures code quality and prevents regression, a requirement for enterprise-grade deployments.

## Future Evolution
While the current system is production-ready, future iterations will focus on:
* *Feature Store Integration:* Transitioning from in-memory feature engineering to a dedicated feature store (e.g., Hopsworks or Feast) to reduce computation latency.
* *Ensemble Methods:* Experimenting with Gradient Boosting (XGBoost/LightGBM) to capture more complex non-linear patterns, using our current Logistic Regression as the baseline challenger.
* *Advanced Drift Mitigation:* Moving from drift detection to automatic retraining triggers via MLflow and automated pipelines.

## Conclusion
This system demonstrates a functional, end-to-end pipeline that transforms raw market data into actionable intelligence. By prioritizing operational excellence alongside machine learning, we have created a service that is both reliable and extensible.
