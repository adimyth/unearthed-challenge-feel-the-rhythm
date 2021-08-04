import pandas as pd  # type: ignore

wa_hols = [
    "2009-01-01",
    "2009-01-26",
    "2009-03-02",
    "2009-04-10",
    "2009-04-11",
    "2009-04-12",
    "2009-04-13",
    "2009-04-25",
    "2009-04-27",
    "2009-06-01",
    "2009-09-28",
    "2009-12-25",
    "2009-12-26",
    "2009-12-28",
    "2010-01-01",
    "2010-01-26",
    "2010-03-01",
    "2010-04-02",
    "2010-04-05",
    "2010-04-26",
    "2010-06-07",
    "2010-09-27",
    "2010-12-25",
    "2010-12-26",
    "2010-12-27",
    "2010-12-28",
    "2011-01-01",
    "2011-01-26",
    "2011-03-07",
    "2011-04-22",
    "2011-04-25",
    "2011-04-26",
    "2011-06-06",
    "2011-10-28",
    "2011-12-25",
    "2011-12-26",
    "2011-12-27",
    "2012-01-01",
    "2012-01-02",
    "2012-01-26",
    "2012-03-05",
    "2012-04-06",
    "2012-04-09",
    "2012-04-25",
    "2012-06-04",
    "2012-10-01",
    "2012-12-25",
    "2012-12-26",
    "2013-01-01",
    "2013-01-26",
    "2013-03-04",
    "2013-03-29",
    "2013-04-01",
    "2013-04-25",
    "2013-06-03",
    "2013-09-30",
    "2013-12-25",
    "2013-12-26",
    "2014-01-01",
    "2014-01-27",
    "2014-03-03",
    "2014-04-18",
    "2014-04-19",
    "2014-04-21",
    "2014-04-25",
    "2014-06-02",
    "2014-09-29",
    "2014-12-25",
    "2014-12-26",
    "2015-01-01",
    "2015-01-26",
    "2015-03-02",
    "2015-04-03",
    "2015-04-04",
    "2015-04-06",
    "2015-04-25",
    "2015-04-27",
    "2015-06-01",
    "2015-09-28",
    "2015-12-25",
    "2016-01-01",
    "2016-01-26",
    "2016-03-07",
    "2016-03-25",
    "2016-03-28",
    "2016-04-25",
    "2016-06-06",
    "2016-09-26",
    "2016-12-25",
    "2016-12-26",
    "2016-12-27",
    "2017-01-01",
    "2017-01-02",
    "2017-01-26",
    "2017-03-06",
    "2017-04-14",
    "2017-04-17",
    "2017-04-25",
    "2017-06-05",
    "2017-09-25",
    "2017-12-25",
    "2017-12-26",
    "2018-01-01",
    "2018-01-26",
    "2018-03-05",
    "2018-03-30",
    "2018-04-02",
    "2018-04-25",
    "2018-06-04",
    "2018-09-24",
    "2018-12-25",
    "2018-12-26",
    "2019-01-01",
    "2019-01-28",
    "2019-03-04",
    "2019-04-19",
    "2019-04-22",
    "2019-04-25",
    "2019-06-03",
    "2019-09-30",
    "2019-12-25",
    "2019-12-26",
    "2020-01-01",
    "2020-01-27",
    "2020-03-02",
    "2020-04-10",
    "2020-04-13",
    "2020-04-25",
    "2020-04-27",
    "2020-06-01",
    "2020-09-28",
    "2020-12-25",
    "2020-12-26",
    "2020-12-28",
    "2021-01-01",
    "2021-01-26",
    "2021-03-01",
    "2021-04-02",
    "2021-04-05",
    "2021-04-25",
    "2021-04-26",
    "2021-06-07",
    "2021-09-27",
    "2021-12-25",
    "2021-12-26",
    "2021-12-27",
    "2021-12-28",
]
wa_hols = pd.to_datetime(wa_hols, format="%Y-%m-%d")

input_cols = [
    "WORK_DESC",
    "TIME_TYPE",
    "FUNC_CAT",
    "TOT_BRK_TM",
    "hour",
    "day_of_week",
    "month",
    "year",
    "holiday",
    "period",
    "season",
    "gap",
]


def categorise_work_desc(data, pat):
    temp = data.loc[data["WORK_DESC"].str.contains(pat)]
    data.loc[temp.index, "WORK_DESC"] = pat
    return data


def add_time_feat(data):
    data["hour"] = data["Work_DateTime"].dt.hour
    data["year"] = data["Work_DateTime"].dt.year
    data["month"] = data["Work_DateTime"].dt.month
    data["date"] = data["Work_DateTime"].dt.date
    data["day_of_week"] = data["Work_DateTime"].dt.dayofweek
    return data


def oversample(df, y_col, n=None, random_state=42):
    """Sample an equal amount from each class, with replacement"""
    gs = [g for _, g in df.groupby(y_col)]
    if n is None:
        n = max(len(g) for g in gs)

    # sample equal number of each group
    gs = [g.sample(n, random_state=random_state, replace=True) for g in gs]
    # concat, and shuffle
    df = pd.concat(gs, 0)
    return df


def preprocess(data_file, is_training=True):
    # Load Data
    df = pd.read_csv(data_file)

    # Add Time Features
    df["Work_DateTime"] = pd.to_datetime(df["Work_DateTime"], errors="coerce")
    df = add_time_feat(df)

    # Categorising WORK_DESC
    df["WORK_DESC"] = df["WORK_DESC"].str.strip().str.upper()
    df = categorise_work_desc(df, "DESIGN CONNECTION ASSETS")
    df = categorise_work_desc(df, "NETWORK PLANNING")
    df = categorise_work_desc(df, "DESIGN INTERNAL")
    df = categorise_work_desc(df, "REPLACE WOOD POLE PWOD/PINT")
    df = categorise_work_desc(df, "DESIGN SYSTEM ASSETS")
    df = categorise_work_desc(df, "REPAIR EARTH PWOD/PINT")
    df = categorise_work_desc(df, "PROJECT PLANNING")
    df = categorise_work_desc(df, "REINFORCE WOOD POLE")
    df = categorise_work_desc(df, "REPLACE OVERHEAD SERVICE")
    df = categorise_work_desc(df, "REPAIR EARTH")
    df = categorise_work_desc(df, "REMOVE WEEDS")
    df = categorise_work_desc(df, "NETWORK FULL INSPECTION")
    df.loc[
        df["WORK_DESC"].str.contains(pat=r"(?=.*REPLACE)(?=.*TYRE)", regex=True),
        "WORK_DESC",
    ] = "REPLACE TYRES"
    df.loc[
        df["WORK_DESC"].str.contains(pat=r"(?=.*REPLACE)(?=.*POLE)", regex=True),
        "WORK_DESC",
    ] = "REPLACE POLE"

    # Considering top few hundred categories
    work_desc_top = df["WORK_DESC"].value_counts().head(500).index.tolist()
    df["WORK_DESC"] = df["WORK_DESC"].apply(
        lambda x: x if x in work_desc_top else "OTHER"
    )

    # Western Australia public holiday
    df["holiday"] = df["Work_DateTime"].dt.round("1D").isin(wa_hols)

    # Period: {"Late Night": 1, "Early Morning": 2, "Morning": 2, "Noon": 3, "Evening": 4, "Night": 5}
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ["Late Night", "Early Morning", "Morning", "Noon", "Evening", "Night"]
    df["period"] = pd.cut(
        df["Work_DateTime"].dt.hour, bins=bins, labels=labels, include_lowest=True
    )

    # Season - {"Summer": 1, "Autumn": 2, "Winter": 3, "Spring": 4}
    df.loc[df["month"].isin([12, 1, 2]), "season"] = 1
    df.loc[df["month"].isin([3, 4, 5]), "season"] = 2
    df.loc[df["month"].isin([6, 7, 8]), "season"] = 3
    df.loc[df["month"].isin([9, 10, 11]), "season"] = 4

    # Gap Between Working Days
    df["gap"] = df.groupby("EmpNo_Anon")["date"].diff().dt.days

    if is_training:
        return df[input_cols + ["incident"]]
    else:
        return df[input_cols]
