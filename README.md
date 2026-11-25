# Project F1: Moodle Activity

**Longitudinal Changes in Moodle Activity and Their Predictive Power for Student Results**

Authors: Marina Gorelikova, Denis Artemjev, Ander Hinn

Topic proposed by Reimo Palm (University of Tartu). LTAT.02.002 Introduction to Data Science, autumn 2026.

## Motivation

We use Moodle logs of the course Theoretical Computer Science (University of Tartu) from the spring semesters 2019 to 2026. In these years teaching changed a lot. In 2020 and 2021 the course was remote because of COVID, then hybrid, and from 2025 in-person again. At the end of 2022 ChatGPT appeared. This makes the data a natural experiment, we can see how students and teachers used Moodle before, during and after these changes.

About 15% of students fail the course or don't come to the exam. If teachers could see early which students are at risk, they could offer help in time.

## Goals

1. Describe how student and teacher activity in Moodle changed from 2019 to 2026.
2. Predict one week before the first midterm which students will fail the course, so that teachers have time to help them.
3. Check if a model trained on older years (2019, 2021 to 2024) still works on new years (2025 and 2026).

## What is in which file

| File | What it does |
|---|---|
| `src/prepare.py` | Shared code. `build_features()` builds the table that the models use: one row is one student in one year, with the target and 14 features. Only data from before the cutoff (one week before the first midterm) is used. |
| `hw10/01_data_prep_denis.ipynb` | Data preparation: how the data was cleaned and which features were made. |
| `hw10/02_trends_marina.ipynb` | Goal 1: how student and teacher activity changed from 2019 to 2026. |
| `hw10/03_models_ander.ipynb` | Goal 2: baseline rule and models that predict who will fail. |
| `results/figures/` | Figures as PDF files. |
| `requirements.txt` | Python packages needed to run the notebooks. |
| `IDS2026_HW07.ipynb` | Homework 7: business understanding and first look at the data. |

## Data

The data is not in this repository, because it is about real students. The logs are pseudonymised, but we still don't publish them. The dataset was given to the students of the course by Reimo Palm (University of Tartu). Anyone who has access to it can repeat the analysis. Put the data into a folder `data/` in the root of the repository like this:

```
data/
  2019/events.parquet, grades.csv, calendar.csv
  2020/...
  ...
  2026/...
  course_runs.csv
```

The folder `data/` is in `.gitignore`, so it is never uploaded to GitHub.

## How to repeat the analysis

1. Clone the repository:

   ```
   git clone https://github.com/anderhinn/Project-F1-Moodle_Activity.git
   ```

2. Install the packages (Python 3.11):

   ```
   pip install -r requirements.txt
   ```

3. Put the data into the `data/` folder as shown above.

4. Open the notebooks in the `hw10/` folder and run them in this order, each with Restart & Run All: `01_data_prep_denis.ipynb`, `02_trends_marina.ipynb`, `03_models_ander.ipynb`. The figures are saved to `results/figures/`.

If `read_parquet` gives an error, check that `pyarrow` is installed. With NumPy 1.x you need `pyarrow` older than version 18.