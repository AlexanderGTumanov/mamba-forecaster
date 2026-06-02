# Beating DLinear with a Mamba forecaster

## Introduction

A well-known result in time series forecasting is that DLinear, a single linear layer per input data channel with no nonlinearity, matches or outperforms most modern Transformer-based models (Zeng et al., 2023). The main implication of this is that most of those models are learning linear trends that a direct regression could capture just as well.

State space models (SSMs) offer a valuable alternative. In particular, the Mamba model family (Gu and Dao, 2023, 2024) is a general-purpose sequence model that has shown strong results on time series forecasting. Similar to the old RNN and LSTM architectures, they maintain a hidden state that evolves over time, but make this state transitions input-dependent, so the model learns what to remember and what to discard at each step.

Using this model makes it possible to beat DLinear even on a limited hardware of a personal laptop. This project implements a Mamba-style SSM from scratch and trains it to outperform DLinear on the ETTm2 benchmark (Electricity Transformer Temperature), running entirely on a 32 GB MacBook M1. The model succeeds at long forecast horizons (L = H = 336 and L = H = 720), where nonlinear dynamics give the model a genuine advantage over linear extrapolation.

---
 
## What this project does

- Retrieves and caches the ETTm2 dataset, normalizes it, and splits it into train, validation, and test sets.
- Implements DLinear: a per-channel linear projection from context window to forecast horizon
- Implements a Mamba-style SSM from scratch, along with RevIN normaliztion layer.
- Trains both models at various benchmark horizon settings and compares their MSE losses on validation and test sets.
