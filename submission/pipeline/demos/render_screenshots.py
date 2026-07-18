# 将 Demo 报告渲染为 HTML 和 PNG。

from __future__ import annotations

import html
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = Path(__file__).resolve().parent


# 生成 Demo 的 HTML 页面和截图。
def main() -> None:
    for scenario in sorted(DEMO_ROOT.glob("scenario_*")):
        if not scenario.is_dir():
            continue
        summary = json.loads((scenario / "summary.json").read_text(encoding="utf-8"))
        report = json.loads((scenario / "output.json").read_text(encoding="utf-8"))
        payload = json.dumps(report, ensure_ascii=False, indent=2)
        route_rows = "".join(
            f"<tr><td>{html.escape(str(item['image']))}</td><td>{html.escape(str(item['worker']))}</td>"
            f"<td>{html.escape(str(item.get('expected_worker', '-')))}</td>"
            f"<td>{'YES' if item.get('routing_error') else 'no'}</td>"
            f"<td>{item['latency_ms']:.3f}</td><td>{html.escape(', '.join(item['gate_reasons']))}</td></tr>"
            for item in report["routing_log"]
        )
        page = f"""<!doctype html><meta charset='utf-8'><title>{html.escape(summary['title'])}</title>
<style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;background:#f5f7fb;color:#172033;margin:30px}}
.card{{background:#fff;border:1px solid #dce2ee;border-radius:12px;padding:22px;margin:14px 0;box-shadow:0 2px 8px #17203312}}
h1{{margin:0 0 8px;color:#173b72}} h2{{color:#275da8}} .score{{font-size:32px;font-weight:bold;color:#17844b}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #e5e9f2;padding:8px;text-align:left;font-size:14px}}
th{{background:#edf3fc}} pre{{white-space:pre-wrap;font-size:11px;line-height:1.35;max-height:560px;overflow:hidden}}
.tag{{display:inline-block;background:#e8f1ff;color:#215b9a;border-radius:12px;padding:4px 10px;margin-right:8px}}
</style><body><div class='card'><h1>{html.escape(summary['title'])}</h1>
<span class='tag'>request_id: {html.escape(str(summary['request_id']))}</span>
<span class='tag'>Mock</span><span class='score'>score {summary['overall_score']}</span></div>
<div class='card'><h2>Routing log</h2><table><tr><th>image</th><th>actual</th><th>expected</th><th>routing error</th><th>latency (ms)</th><th>gate reasons</th></tr>{route_rows}</table></div>
<div class='card'><h2>Findings / compliance</h2><pre>{html.escape(json.dumps({'findings':report['findings'],'compliance_items':report['compliance_items'],'recommendations':report['recommendations']},ensure_ascii=False,indent=2))}</pre></div>
<div class='card'><h2>Full output JSON</h2><pre>{html.escape(payload)}</pre></div></body>"""
        target = Path("/tmp") / f"task3-{scenario.name}.html"
        target.write_text(page, encoding="utf-8")
        chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
        if not chrome:
            raise RuntimeError("google-chrome or google-chrome-stable is required to render screenshots")
        screenshot = scenario / "screenshot.png"
        command = [
            chrome,
            "--headless",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--window-size=1280,1800",
            f"--screenshot={screenshot}",
            target.as_uri(),
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            if not screenshot.is_file() or screenshot.stat().st_size == 0:
                raise
        print(f"{scenario.name}\t{screenshot}")


if __name__ == "__main__":
    main()
