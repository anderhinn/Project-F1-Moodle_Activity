"""
Data preparation for project F1 (Moodle Activity).

build_features(data_dir) turns the raw Moodle data into one table:
one row = one student in one year.

Columns of the result:
  year, actor_id        - who and when (identifiers, not features)
  fail                  - TARGET: 1 = F or no grade, 0 = A-E
  is_repeat             - 1 if the student already took the course in an earlier year
  active_days, active_weeks, active_weeks_share,
  first_day_lag, active_days_last_2_weeks
                        - general activity before the cutoff
  days_materials, days_homework, days_lectures, days_forum, days_grades
                        - on how many days the student used each group of Moodle tools
  hw_early_count, hw_early_submitted_share, hw_early_mean
                        - early homeworks (deadline before the cutoff)

Cutoff = first midterm minus 7 days. Only data from before the cutoff is used,
because the prediction must be made one week before the midterm.
"""

import pandas as pd

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Moodle tools changed over the years (e.g. homework was a Quiz in 2021-2022
# and an Assignment in other years), so we group components into general groups.
COMPONENT_GROUPS = {
    "File": "materials",
    "URL": "materials",
    "Folder": "materials",
    "Page": "materials",
    "Assignment": "homework",
    "Quiz": "homework",
    "File submissions": "homework",
    "Online text submissions": "homework",
    "BigBlueButton": "lectures",
    "BigBlueButtonBN": "lectures",
    "Forum": "forum",
    "User report": "grades",
    "Overview report": "grades",
}


def find_year_folders(data_dir):
    """The year folders can be in data/2019/ or, if the whole dataset zip was unpacked
    into data/, in data/data/2019/. Return the folder that contains them."""
    import os
    if os.path.isdir(data_dir + "/2019"):
        return data_dir
    if os.path.isdir(data_dir + "/data/2019"):
        return data_dir + "/data"
    raise FileNotFoundError("No year folders (2019, 2020, ...) found in " + data_dir)


def load_raw(data_dir):
    """Read events, grades and calendar of all years into three tables."""
    data_dir = find_year_folders(data_dir)
    events_list = []
    grades_list = []
    calendar_list = []
    for year in YEARS:
        folder = data_dir + "/" + str(year) + "/"
        events_list.append(pd.read_parquet(folder + "events.parquet"))
        grades_list.append(pd.read_csv(folder + "grades.csv"))
        calendar_list.append(pd.read_csv(folder + "calendar.csv"))

    events = pd.concat(events_list, ignore_index=True)
    grades = pd.concat(grades_list, ignore_index=True)
    calendar = pd.concat(calendar_list, ignore_index=True)

    events["year"] = events["course_run_id"].str[-4:].astype(int)
    grades["year"] = grades["course_run_id"].str[-4:].astype(int)
    calendar["year"] = calendar["course_run_id"].str[-4:].astype(int)

    # Calendar times are text: convert them to real times (in UTC, like the events)
    calendar["start_time"] = pd.to_datetime(calendar["start_time"], utc=True)
    calendar["end_time"] = pd.to_datetime(calendar["end_time"], utc=True)
    events["event_time"] = events["event_time"].dt.tz_convert("UTC")
    return events, grades, calendar


def get_cutoffs(calendar):
    """Cutoff of every year = start of the first midterm minus 7 days."""
    midterm = calendar[calendar["activity_name"] == "midterm_attempt_1"]
    cutoff = midterm.set_index("year")["start_time"] - pd.Timedelta(days=7)
    return cutoff


def merge_languages(grades):
    """Every task has an _eng and an _est column. Keep one column per task (the higher score)."""
    tasks = []
    for col in grades.columns:
        if col.endswith("_eng"):
            tasks.append(col[:-4])
    for task in tasks:
        grades[task] = grades[[task + "_eng", task + "_est"]].max(axis=1)
        grades = grades.drop(columns=[task + "_eng", task + "_est"])
    return grades


def activity_features(events, cutoff):
    """Activity of every student from the start of the course until the cutoff."""
    students = events[events["actor_id"] != "T"].copy()
    students["cutoff"] = students["year"].map(cutoff)
    students = students[students["event_time"] < students["cutoff"]]

    # Day and week are counted in Estonian time
    local_time = students["event_time"].dt.tz_convert("Europe/Tallinn")
    students["date"] = local_time.dt.tz_localize(None).dt.normalize()

    # Moodle opened on different dates in different years, so we count weeks
    # from the first day any student was active that year
    year_start = students.groupby("year")["date"].min()
    students["day_nr"] = (students["date"] - students["year"].map(year_start)).dt.days
    students["week_nr"] = students["day_nr"] // 7

    # How many days / weeks there were until the cutoff (for normalising)
    cutoff_date = cutoff.dt.tz_convert("Europe/Tallinn").dt.tz_localize(None).dt.normalize()
    days_total = (cutoff_date - year_start).dt.days
    weeks_total = days_total // 7 + 1

    by_student = students.groupby(["year", "actor_id"])
    features = pd.DataFrame({
        "active_days": by_student["date"].nunique(),
        "active_weeks": by_student["week_nr"].nunique(),
        "first_day_lag": by_student["day_nr"].min(),
    }).reset_index()
    features["active_weeks_share"] = features["active_weeks"] / features["year"].map(weeks_total)

    # Activity in the last two weeks before the cutoff
    last_two = students[students["day_nr"] > students["year"].map(days_total) - 14]
    recent = last_two.groupby(["year", "actor_id"])["date"].nunique().rename("active_days_last_2_weeks")
    features = features.merge(recent.reset_index(), on=["year", "actor_id"], how="left")

    # On how many days the student used each group of tools
    students["group"] = students["component"].map(COMPONENT_GROUPS)
    group_days = students.dropna(subset=["group"]).groupby(["year", "actor_id", "group"])["date"].nunique()
    group_days = group_days.unstack(fill_value=0).add_prefix("days_").reset_index()
    features = features.merge(group_days, on=["year", "actor_id"], how="left")

    return features, days_total


def homework_features(grades, calendar, cutoff):
    """Scores of the homeworks whose deadline is before the cutoff."""
    homework = calendar[calendar["activity_type"] == "homework"].copy()
    homework["cutoff"] = homework["year"].map(cutoff)
    early = homework[homework["end_time"] < homework["cutoff"]]

    rows = []
    for year in grades["year"].unique():
        hw_names = list(early.loc[early["year"] == year, "activity_name"])
        year_grades = grades[grades["year"] == year]
        scores = year_grades[hw_names]
        rows.append(pd.DataFrame({
            "year": year,
            "actor_id": year_grades["actor_id"],
            "hw_early_count": len(hw_names),
            # share of early homeworks that the student submitted (has a score)
            "hw_early_submitted_share": scores.notna().sum(axis=1) / len(hw_names),
            # mean score, a missing homework counts as 0 points
            "hw_early_mean": scores.fillna(0).mean(axis=1),
        }))
    return pd.concat(rows, ignore_index=True)


def build_features(data_dir, keep_2020=False):
    """Main function: returns the table of features + target, one row per student per year."""
    events, grades, calendar = load_raw(data_dir)
    cutoff = get_cutoffs(calendar)
    grades = merge_languages(grades)

    # Start from the grades table: it contains every student, also those with no Moodle events
    table = grades[["year", "actor_id", "course_letter_grade"]].copy()

    # TARGET: fail = F or no grade (did not show up). 2020 has no grades at all.
    table["fail"] = table["course_letter_grade"].isna() | (table["course_letter_grade"] == "F")
    table["fail"] = table["fail"].astype(int)
    table.loc[table["year"] == 2020, "fail"] = None
    table = table.drop(columns=["course_letter_grade"])

    # Did the student take the course already in an earlier year?
    first_year = table.groupby("actor_id")["year"].transform("min")
    table["is_repeat"] = (table["year"] > first_year).astype(int)

    activity, days_total = activity_features(events, cutoff)
    table = table.merge(activity, on=["year", "actor_id"], how="left")

    # No events before the cutoff = 0 activity
    count_cols = [c for c in table.columns if c.startswith("active_") or c.startswith("days_")]
    table[count_cols] = table[count_cols].fillna(0)
    # Never active: first activity "at the cutoff"
    table["first_day_lag"] = table["first_day_lag"].fillna(table["year"].map(days_total))

    homework = homework_features(grades, calendar, cutoff)
    table = table.merge(homework, on=["year", "actor_id"], how="left")

    # Counts of days are whole numbers
    int_cols = ["active_days", "active_weeks", "first_day_lag", "active_days_last_2_weeks"]
    int_cols += [c for c in table.columns if c.startswith("days_")]
    table[int_cols] = table[int_cols].astype(int)

    if not keep_2020:
        table = table[table["year"] != 2020].copy()
        table["fail"] = table["fail"].astype(int)
    return table.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    df = build_features(data_dir)
    print(df.shape)
    print(df.head())
