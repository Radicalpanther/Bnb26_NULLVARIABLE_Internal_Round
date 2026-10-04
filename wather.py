# watcher.py - tails an agent's JSONL log, diagnoses failed runs, and raises an alert.
# Usage: .venv/bin/python watcher.py agent.log.jsonl [--from-start]
import argparse, json, os, time
import requests

API = "http://localhost:8000/diagnose?llm=false"   # template explanation: fast and always correct
ALERT_URL = None        # e.g. "http://localhost:3000/alerts" once your backend teammate adds it
IDLE_SECONDS = 30       # no new lines for this long = the agent crashed, count it as failed

runs = {}               # run_id -> {"steps": [...], "last_seen": t, "status": None, "done": False}

def handle_line(line):
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return                                   # ignore half-written or junk lines
    rid = ev.get("run_id")
    if not rid:
        return
    r = runs.setdefault(rid, {"steps": [], "last_seen": 0, "status": None, "done": False})
    r["last_seen"] = time.time()
    if ev.get("type") == "run_end":
        r["status"] = ev.get("status")
    else:
        r["steps"].append(ev)

def check_finished():
    for rid, r in runs.items():
        if r["done"]:
            continue
        ended = r["status"] is not None or time.time() - r["last_seen"] > IDLE_SECONDS
        if not ended:
            continue
        r["done"] = True
        if r["status"] == "success":
            continue
        diagnose_and_alert(rid, r["steps"])

def diagnose_and_alert(rid, steps):
    try:
        resp = requests.post(API, json={"run_id": rid, "steps": steps}, timeout=60).json()
        top3 = resp["ranking"][:3]
    except Exception as e:
        print(f"[watcher] could not get a diagnosis for {rid}: {e}")
        return
    top = top3[0]
    print("\n" + "=" * 64)
    print(f"ALERT: agent run {rid} failed")
    print(f"Most likely cause: step {top['step_index']} ({top['step_type']}), score {top['score']}")
    print("Evidence:", "; ".join(top["evidence"]))
    print("Other suspects:", ", ".join(f"step {s['step_index']}" for s in top3[1:]))
    print("Explanation:", resp["explanation"])
    print("=" * 64 + "\n")
    if ALERT_URL:
        try:
            requests.post(ALERT_URL, json={"run_id": rid, "diagnosis": resp}, timeout=10)
        except Exception as e:
            print(f"[watcher] could not post the alert to the backend: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from-start", action="store_true")
    args = ap.parse_args()
    while not os.path.exists(args.log):          # wait until the agent creates the log
        time.sleep(0.5)
    print(f"[watcher] watching {args.log}")
    buf = ""
    with open(args.log, "r", encoding="utf-8") as f:
        if not args.from_start:
            f.seek(0, 2)                         # skip old content, only watch new lines
        while True:
            chunk = f.readline()
            if chunk:
                buf += chunk
                if buf.endswith("\n"):           # only handle complete lines
                    handle_line(buf)
                    buf = ""
            else:
                check_finished()
                time.sleep(0.5)

if __name__ == "__main__":
    main()