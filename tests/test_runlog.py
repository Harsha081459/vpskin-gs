import csv
import json

from utils.runlog import RunLog


def test_runlog_writes_row(tmp_path):
    p = tmp_path / "runs.csv"
    with RunLog("smoke", config={"a": 1}, csv_path=str(p)) as rl:
        rl.log(psnr=30.0)
    with RunLog("smoke2", csv_path=str(p)):
        pass
    rows = list(csv.DictReader(open(p)))
    assert [r["run"] for r in rows] == ["smoke", "smoke2"]
    assert json.loads(rows[0]["metrics"]) == {"psnr": 30.0}
    assert rows[0]["status"] == "ok" and float(rows[0]["wall_clock_s"]) >= 0
