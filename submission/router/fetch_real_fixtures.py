# 下载 Wikimedia Commons 许可图片作为路由夹具。

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
FIXTURES = ROOT / "fixtures"
API = "https://commons.wikimedia.org/w/api.php"

CATEGORIES = {
    "a": "Category:Supermarket aisles",
    "b": "Category:Street photography",
    "c": "Category:Paper documents",
    "d": "Category:Emergency exits",
}
COUNTS = {"a": 19, "b": 19, "c": 19, "d": 18}


# 请求 Wikimedia Commons API。
def api(params: dict) -> dict:
    query = urlencode({"action": "query", "format": "json", **params})
    request = Request(
        f"{API}?{query}",
        headers={
            "User-Agent": "Task2FixtureCollector/1.0 (educational evaluation; contact local repo owner)"
        },
    )
    with urlopen(request, timeout=40) as response:
        return json.load(response)


# 获取分类下的可用图片及许可信息。
def files_in_category(category: str, count: int) -> list[dict]:
    selected: list[dict] = []
    continuation: dict = {}
    seen: set[str] = set()
    while len(selected) < count:
        payload = api(
            {
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": "800",
                **continuation,
            }
        )
        for page in payload.get("query", {}).get("pages", {}).values():
            title = page.get("title", "")
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if (
                not mime.startswith("image/")
                or mime in {"image/svg+xml", "image/gif"}
                or title in seen
            ):
                continue
            metadata = info.get("extmetadata", {})
            license_name = metadata.get("LicenseShortName", {}).get("value", "").strip()
            if not license_name:
                continue
            selected.append(
                {
                    "title": title,
                    "url": info.get("thumburl") or info.get("url"),
                    "original_url": info.get("url"),
                    "description_url": info.get("descriptionurl"),
                    "license": re.sub(r"<[^>]+>", "", license_name),
                    "author": re.sub(
                        r"<[^>]+>", "", metadata.get("Artist", {}).get("value", "")
                    ).strip(),
                    "category": category,
                }
            )
            seen.add(title)
            if len(selected) == count:
                return selected
        continuation = payload.get("continue", {})
        if not continuation:
            break
    if len(selected) < count:
        raise RuntimeError(
            f"{category}: found {len(selected)} usable images, need {count}"
        )
    return selected


# 下载单张图片并处理限流重试。
def download(url: str, target: Path) -> None:
    for attempt in range(6):
        try:
            request = Request(url, headers={"User-Agent": "Task2FixtureCollector/1.0"})
            with urlopen(request, timeout=60) as response, target.open("wb") as output:
                output.write(response.read())
            return
        except HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            wait = min(60, 10 * (attempt + 1))
            print(f"rate limited; waiting {wait}s", flush=True)
            time.sleep(wait)
        except URLError:
            if attempt == 5:
                raise
            wait = min(60, 10 * (attempt + 1))
            print(f"network retry; waiting {wait}s", flush=True)
            time.sleep(wait)


# 下载全部图片并写入来源清单。
def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for prefix, category in CATEGORIES.items():
        items = files_in_category(category, COUNTS[prefix])
        for index, item in enumerate(items):
            filename = f"{prefix}_{index:02d}.jpg"
            target = FIXTURES / filename
            if not target.exists() or target.stat().st_size == 0:
                download(item["url"], target)
            manifest.append({"image_path": str(Path("fixtures") / filename), **item})
            print(filename, item["title"])
            time.sleep(0.15)
    with (ROOT / "real_fixture_sources.jsonl").open("w", encoding="utf-8") as output:
        for record in manifest:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"downloaded {len(manifest)} real photos")


if __name__ == "__main__":
    main()
