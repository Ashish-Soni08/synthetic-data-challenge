# THE MOSTLY AI PRIZE

[Leaderboard](https://www.mostlyaiprize.com/leaderboard)

- Flat Leaderboard: `17th`
- Sequential Leaderboard: `11th`

**Timeline:** May 14, 2025 - Jul 3, 2025

---

## Competition Overview

The goal of this challenge is to generate the best synthetic data.

There are two independent challenges:

> The task is to generate a novel synthetic dataset that matches the structure of a provided dataset and preserves its statistical patterns, while its records are NOT significantly closer to the (released) original samples than to the (unreleased) holdout samples.

**NOTE:** To succeed, you'll typically train a generative model on the training data, ensuring it generalizes well without overfitting.

### THE FLAT DATA Challenge

> 100,000 records, with 80 data columns: 60 numeric, 20 categorical.

**NOTE:** Any submission to the FLAT DATA challenge needs to be in CSV format, contain 100,000 records, and consists of the same columns, to be valid.

### THE SEQUENTIAL DATA Challenge

> 20,000 groups, with 5-10 records each, with 10 data columns: 7 numeric, 3 categorical.

**NOTE:** A submission to the SEQUENTIAL DATA challenge needs to be in CSV format, contain 20,000 groups, and consists of the same columns, to be valid. Note that you can pick any format for the group ID column. That column will only be used to group the corresponding events together.

### Evaluation Criteria

To qualify for the Leaderboard, a submission must meet two privacy thresholds: 

- DCR Share below 52%
- NNDR Ratio above 0.5

These metrics ensure that the generated samples are sufficiently distinct — just as far from the training data as from the holdout set — while still capturing the core statistical structure.

The final evaluations will take place on AWS EC2 instances (c5d.12xlarge for CPU runs and g5.2xlarge for GPU runs), with a strict time cap of six hours per execution.

Final evaluations will take into account five factors:

- accuracy
- privacy
- ease of use
- compute efficiency
- generalizability

---

## Experiments

I tried different batch sizes for the MostlyAI models for both datasets. The best-peforming models for synthetic data generation are:

### Flat Data Challenge

- **Batch Size:** `1024`
- **Model:** `MOSTLY_AI/Large`
- **Accuracy:** `98.1`
- **DCR Share:** `51.0`
- **NNDR Ratio:** `1.024`
- **Training Checkpoint Epoch:** `100`
- **Validation loss:** `110.672`
- **Samples:** `7.89M`
- **[Model Report](https://app.mostly.ai/api/v2/generators/fc84c8ce-5b1d-4e4e-a53d-0284a52b87e0/tables/a5f1b62a-054f-497c-a905-5af364b9ca9f/report?modelType=TABULAR&slft=LA5c2O5xNdebEhrXEnXWwg0_zO379vFlPRBbITJb1CauXBMNMFoMoFXoJtMLldrmlfcr9LCMRxGT4avxx0HanoNhMxGj2g9FMlrFB9Y28a1QcLCoQWmCMq7sfuGOyUxv)**

### Sequential Data Challenge

- **Batch Size:** `512`
- **Model:** `MOSTLY_AI/Medium`
- **Accuracy:** `96`
- **DCR Share:** `51.8`
- **NNDR Ratio:** `1.244`
- **Training Checkpoint Epoch:** `41`
- **Validation loss:** `14.397`
- **Samples:** `633.04K`
- **[Model Report](https://app.mostly.ai/api/v2/generators/1f850b77-c823-4f90-9916-5557fa4c42b7/tables/e0303753-5e66-4a9c-a02e-1d84fa07546f/report?modelType=TABULAR&slft=LA5c2O5xNdebEhrXEnXWwg0_zO379vFlPRBbITJb1CbanVlTAh0gYXipOS04wUDgh7tv5W883trQvwqGYBg7M62AZaMbmkYLVKx1YfAzsxNmS6gDA8maUnVjyzWC7tQD)**

---

## Training Environment

- **Datasets**: flat-training.csv (100,000 rows, 80 columns), sequential-training.csv (20,000 groups, 5-10 records each, 10 columns)
- **Hardware**: NVIDIA A10/A10G GPU, 673-906 GB RAM, 17 CPUs on [Modal](https://modal.com)
- **SDK**: MostlyAI v4.7.9 (LOCAL mode)

---

## References

- [Tutorial](https://www.youtube.com/watch?v=SGrQKFR2Mvg)
- [Flat Data Challenge Gist](https://gist.github.com/mplatzer/1a30319a5e9fb2e560b4fa51f776cab7)
- [Sequential Data Challenge Gist](https://gist.github.com/mplatzer/552a28bb8acc1e951541bea8fb3ebbd1)