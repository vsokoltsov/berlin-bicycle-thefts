from typing import List
import os
import json
import click
import lightgbm as lgb
from pydantic import TypeAdapter

from .models import PredictItem
from .predictor import Predictor
from .config import PROCESSED_DATA, MODEL_PATH


@click.command()
@click.argument("items", nargs=1)
@click.option(
    "--force",
    default=False,
    help="""
force to load dataset.
Default: false.
If dataset was already saved - returns file on disc
""",
)
def predict(items: str, force: bool) -> None:
    adapter = TypeAdapter(List[PredictItem])
    try:
        items_json: List[PredictItem] = adapter.validate_json(items)
    except json.JSONDecodeError:
        raise click.ClickException("Invalid argument format. Expecting json string.Example:'[{\"lor\":\"10100205\",\"date\":\"2025-11-14\"}]'")

    # Load artifacts and model
    with open(os.path.join(PROCESSED_DATA, "preproc.json"), "r") as f:
        artifacts = json.load(f)

    model = lgb.Booster(model_file=MODEL_PATH)

    predictor = Predictor(
        model=model,
        artifacts=artifacts
    )
    df = predictor.build(items_json)
    result = predictor.predict(df)

    out = {
        "model_version": os.path.basename(MODEL_PATH),
        "feature_version": "v3",
        "items": [
            {
                "lor": item.lor,
                "date": item.date.strftime("%Y-%m-%d"),
                "rate_hat_per_1000": item.rate_hat_per_1000,
                "count_hat": item.count_hat,
                "count_int": item.count_int
            }
            for item in result
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    predict()