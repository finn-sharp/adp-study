"""고정 seed에서 한 번 생성하는 모의16~20회 독립 합성 자료. 실제 기출 데이터 아님."""

from pathlib import Path
import numpy as np
import pandas as pd


def generate(folder, spec):
    out = Path(folder) / "data"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(spec["seed"])

    def save(name, frame):
        frame.to_csv(out / name, index=False, float_format="%.12g")

    for sec in spec["sections"]:
        family = sec["family"]
        cfg = sec["settings"]
        if family == "reg":
            x = rng.normal(size=(240, 3))
            group = rng.choice(["A", "B"], 240)
            y = (
                100
                + 8 * x[:, 0]
                - 5 * x[:, 1]
                + 3 * x[:, 2]
                + 4 * (group == "B")
                + rng.normal(0, 3, 240)
            )
            if cfg["interaction"]:
                y += 9 * x[:, 0] * x[:, 1]
            x[rng.choice(240, 6, replace=False), 0] = np.nan
            frame = pd.DataFrame(x, columns=["x1", "x2", "x3"])
            frame.insert(0, "id", np.arange(240) + 1)
            frame["group"] = group
            frame["target"] = y
            save("regression.csv", frame)
        elif family == "cls":
            x = rng.normal(size=(384, 3))
            g = rng.choice(["A", "B", "C"], 384)
            z = 1.2 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2] + 0.4 * (g == "C")
            if cfg["classes"] == 2:
                y = (rng.uniform(size=384) < 1 / (1 + np.exp(-(-1.4 + z)))).astype(int)
            else:
                scores = np.column_stack(
                    [np.zeros(384), z - 0.1, -z + 0.6 * x[:, 1] - 0.2]
                )
                p = np.exp(scores - scores.max(axis=1)[:, None])
                p /= p.sum(axis=1)[:, None]
                y = np.array([rng.choice(3, p=v) for v in p])
            x[rng.choice(360, 9, replace=False), 0] = np.nan
            frame = pd.DataFrame(x, columns=["x1", "x2", "x3"])
            frame.insert(0, "id", np.arange(384) + 1)
            frame["group"] = g
            train = frame.iloc[:360].copy()
            train["target"] = y[:360]
            save("classification.csv", train)
            new = frame.iloc[360:].copy()
            new.loc[new.index[0], "group"] = "NEW"
            save("submission.csv", new)
        elif family == "text":
            tokens = [
                ("movie", "NOUN", np.nan),
                ("story", "NOUN", np.nan),
                ("actor", "NOUN", np.nan),
                ("sound", "NOUN", np.nan),
                ("great", "ADJ", 1),
                ("clear", "ADJ", 1),
                ("dull", "ADJ", -1),
                ("weak", "ADJ", -1),
                ("average", "ADJ", 0),
                ("the", "STOP", np.nan),
                ("is", "STOP", np.nan),
                ("and", "STOP", np.nan),
                ("not", "STOP", np.nan),
            ]
            save(
                "lexicon.csv",
                pd.DataFrame(tokens, columns=["token", "pos", "polarity"]),
            )
            phrases = [
                "The movie is great and the story is clear.",
                "The actor is dull and the sound is weak.",
                "The story is average.",
                "The movie is not great.",
                "unknownword mystery",
                "The movie movie is clear.",
                "",
            ]
            save(
                "reviews.csv",
                pd.DataFrame(
                    dict(
                        id=np.arange(1, 61),
                        group=rng.choice(["A", "B"], 60),
                        text=[phrases[i % len(phrases)] for i in range(60)],
                    )
                ),
            )
        elif family == "assoc":
            baskets = []
            for i in range(100):
                basket = []
                if rng.rand() < 0.65:
                    basket += ["bread"]
                if rng.rand() < (0.8 if "bread" in basket else 0.2):
                    basket += ["milk"]
                if rng.rand() < 0.4:
                    basket += ["fruit"]
                if rng.rand() < 0.3:
                    basket += ["tea"]
                if basket and i % 9 == 0:
                    basket += [basket[0]]
                baskets.append("|".join(basket))
            save("baskets.csv", pd.DataFrame(dict(id=np.arange(1, 101), items=baskets)))
        elif family == "ts":
            p = cfg["period"]
            t = np.arange(96)
            e = rng.normal(0, 1.2, 96)
            for j in range(1, 96):
                e[j] += 0.35 * e[j - 1]
            y = 80 + 0.22 * t + 8 * np.sin(2 * np.pi * t / p) + e
            save(
                "series.csv",
                pd.DataFrame(
                    dict(
                        date=pd.date_range(
                            "2015-01-01", periods=96, freq="MS" if p == 12 else "QS"
                        ),
                        target=y,
                    )
                ),
            )
        elif family == "survey":
            group = rng.choice(["A", "B"], 180)
            latent = rng.normal(0, 0.7, 180) + 0.25 * (group == "B")
            latent2 = rng.normal(0, 0.5, 180)
            x = np.column_stack(
                [
                    latent + rng.normal(0, 0.5, 180),
                    latent + rng.normal(0, 0.5, 180),
                    0.5 * latent + latent2 + rng.normal(0, 0.4, 180),
                    0.5 * latent + latent2 + rng.normal(0, 0.4, 180),
                ]
            )
            x = np.clip(np.round(3 + x), 1, 5)
            x[:, [1, 3]] = 6 - x[:, [1, 3]]
            x[rng.choice(180, 8, False), 0] = np.nan
            frame = pd.DataFrame(x, columns=["q1", "q2", "q3", "q4"])
            frame.insert(0, "group", group)
            frame.insert(0, "id", np.arange(1, 181))
            save("survey.csv", frame)
        elif family == "risk":
            rows = []
            for j, c in enumerate("ABCD"):
                daily = rng.poisson(20 + 5 * j, 70).astype(float)
                if c == "B":
                    daily[35] = -8
                deaths = rng.binomial(2, 0.15 + 0.05 * j, 70)
                for date, cases, deceased in zip(
                    pd.date_range("2020-01-01", periods=70),
                    np.cumsum(daily),
                    np.cumsum(deaths),
                ):
                    rows.append(
                        dict(
                            country=c,
                            date=date,
                            population=(j + 1) * 100000,
                            cases=cases,
                            deaths=deceased,
                        )
                    )
            save("risk.csv", pd.DataFrame(rows))
        elif family == "lag":
            t = np.arange(240)
            e = rng.normal(0, 0.8, 240)
            for j in range(1, 240):
                e[j] += 0.6 * e[j - 1]
            y = 18 + 7 * np.sin(2 * np.pi * t / 365) + 2 * np.sin(2 * np.pi * t / 7) + e
            frame = pd.DataFrame(
                dict(date=pd.date_range("2021-01-01", periods=240), target=y)
            )
            frame["lag1"] = frame.target.shift(1)
            frame["lag2"] = frame.target.shift(2)
            frame.loc[25, "lag1"] += 6
            frame.loc[97, "lag2"] -= 4
            frame["audit"] = y + 0.01 * rng.normal(size=240)
            save("temperature.csv", frame)
        elif family == "energy":
            rows = []
            for h in range(12):
                for stamp in pd.date_range("2021-02-01", periods=14 * 24, freq="h"):
                    hour = stamp.hour
                    peak = (7, 14, 20)[h % 3]
                    value = (
                        0.2
                        + 0.06 * h
                        + 0.8 * np.exp(-0.5 * ((hour - peak) / 2) ** 2)
                        + 0.12 * (stamp.dayofweek >= 5)
                        + rng.uniform(0, 0.08)
                    )
                    rows.append(
                        dict(household="H%02d" % (h + 1), timestamp=stamp, kwh=value)
                    )
            save("energy.csv", pd.DataFrame(rows))
