"""
scrape_images.py — Image scraper for LoRA training data.

Sub-commands:
  instagram   Download images from an Instagram profile/post via Instaloader
  site        Crawl any website with Playwright (handles JS/lazy-load)

Instagram usage:
    uv run scrape_images.py instagram --target <username>
    uv run scrape_images.py instagram --target <username> --posts 50
    uv run scrape_images.py instagram --target <shortcode>  --mode post
    uv run scrape_images.py instagram --target <username> --login --session-file session.json

Site usage:
    uv run scrape_images.py site --url https://murpheys.com
    uv run scrape_images.py site --url https://murpheys.com --depth 3 --out datasets/murpheys
    uv run scrape_images.py site --url https://murpheys.com --static
"""

from __future__ import annotations

import argparse
import io
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── helpers ───────────────────────────────────────────────────────────────────


def normalise_url(url: str) -> str:
    p = urlparse(url)
    return p._replace(fragment="").geturl().rstrip("/")


def is_same_origin(url: str, origin: str) -> bool:
    return urlparse(url).netloc == urlparse(origin).netloc


def looks_like_image(url: str, allowed_exts: set[str]) -> bool:
    path = urlparse(url).path.lower().split("?")[0]
    return any(path.endswith(f".{ext}") for ext in allowed_exts)


def sanitise_filename(url: str) -> str:
    p = urlparse(url)
    name = (p.netloc + p.path).replace("/", "_").replace("?", "_").replace("&", "_")
    name = re.sub(r"[^\w.\-]", "_", name)
    return name[:200]


def extract_images_from_html(html: str, page_url: str, allowed_exts: set[str]) -> list[str]:
    """Parse rendered HTML and return absolute image URLs."""
    soup = BeautifulSoup(html, "html.parser")
    img_srcs: list[str] = []

    # <img> with various lazy-load attributes
    for tag in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-image"):
            src = tag.get(attr, "")
            if src and not src.startswith("data:"):
                img_srcs.append(src)

    # srcset on <img> and <source>
    for tag in soup.find_all(["img", "source"]):
        for attr in ("srcset", "data-srcset"):
            srcset = tag.get(attr, "")
            for part in srcset.split(","):
                tokens = part.strip().split()
                if tokens and not tokens[0].startswith("data:"):
                    img_srcs.append(tokens[0])

    # og:image / twitter:image meta
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") + meta.get("name", "")
        if "image" in prop.lower():
            content = meta.get("content", "")
            if content and not content.startswith("data:"):
                img_srcs.append(content)

    # background-image URLs in style attributes / <style> blocks
    for style_text in [tag.get("style", "") for tag in soup.find_all(style=True)]:
        for match in re.findall(r'url\(["\']?([^"\')\s]+)["\']?\)', style_text):
            if not match.startswith("data:"):
                img_srcs.append(match)

    result: list[str] = []
    seen: set[str] = set()
    for src in img_srcs:
        abs_url = urljoin(page_url, src)
        norm = normalise_url(abs_url)
        if norm not in seen and looks_like_image(abs_url, allowed_exts):
            seen.add(norm)
            result.append(abs_url)
    return result


def extract_links_from_html(html: str, page_url: str, root: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"])
        if is_same_origin(href, root):
            links.append(href)
    return links


# ── Playwright crawler ────────────────────────────────────────────────────────


def crawl_playwright(
    root: str,
    max_depth: int,
    delay: float,
    wait_ms: int,
    scroll: bool,
    allowed_exts: set[str],
) -> list[str]:
    """BFS crawl using headless Chromium — handles JS-rendered pages."""
    from playwright.sync_api import sync_playwright

    visited_pages: set[str] = set()
    found_images: list[str] = []
    seen_images: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(normalise_url(root), 0)])

    # Also intercept network image requests (catches dynamically loaded images)
    intercepted: set[str] = set()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # Intercept every image request the browser makes
        def on_response(response: object) -> None:
            url = response.url  # type: ignore[attr-defined]
            ct = response.headers.get("content-type", "")  # type: ignore[attr-defined]
            if "image" in ct or looks_like_image(url, allowed_exts):
                norm = normalise_url(url)
                if norm not in intercepted:
                    intercepted.add(norm)

        page.on("response", on_response)

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Crawling[/] {task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(root, total=None)

            while queue:
                url, depth = queue.popleft()
                norm = normalise_url(url)
                if norm in visited_pages:
                    continue
                visited_pages.add(norm)
                progress.update(task, description=f"[dim]{url[:70]}[/]")

                try:
                    page.goto(url, wait_until="networkidle", timeout=20_000)
                    page.wait_for_timeout(wait_ms)

                    if scroll:
                        # Scroll incrementally to trigger lazy-load
                        page.evaluate("""
                            async () => {
                                await new Promise(resolve => {
                                    let total = document.body.scrollHeight;
                                    let step = Math.ceil(total / 10);
                                    let pos = 0;
                                    const timer = setInterval(() => {
                                        window.scrollBy(0, step);
                                        pos += step;
                                        if (pos >= total) { clearInterval(timer); resolve(); }
                                    }, 200);
                                });
                            }
                        """)
                        page.wait_for_timeout(1000)

                    html = page.content()
                except Exception as exc:
                    console.log(f"[yellow]Skip[/] {url} — {exc}")
                    continue

                # Images from parsed HTML
                for img_url in extract_images_from_html(html, url, allowed_exts):
                    norm_img = normalise_url(img_url)
                    if norm_img not in seen_images:
                        seen_images.add(norm_img)
                        found_images.append(img_url)

                # Images from DOM src attributes (post-JS evaluation)
                try:
                    js_srcs: list[str] = page.evaluate("""
                        () => Array.from(document.images).map(i => i.currentSrc || i.src)
                              .filter(Boolean)
                    """)
                    for src in js_srcs:
                        abs_url = urljoin(url, src)
                        if not src.startswith("data:"):
                            norm_img = normalise_url(abs_url)
                            if norm_img not in seen_images and looks_like_image(
                                abs_url, allowed_exts
                            ):
                                seen_images.add(norm_img)
                                found_images.append(abs_url)
                except Exception:
                    pass

                if depth < max_depth:
                    for link in extract_links_from_html(html, url, root):
                        if normalise_url(link) not in visited_pages:
                            queue.append((link, depth + 1))

                time.sleep(delay)

        browser.close()

    # Add network-intercepted images
    for norm in intercepted:
        if norm not in seen_images and looks_like_image(norm, allowed_exts):
            seen_images.add(norm)
            found_images.append(norm)

    console.log(
        f"[green]Crawled[/] {len(visited_pages)} pages, "
        f"found [bold]{len(found_images)}[/] images "
        f"([dim]+{len(intercepted)} intercepted[/])"
    )
    return found_images


# ── static (httpx) crawler ────────────────────────────────────────────────────


def crawl_static(
    root: str,
    max_depth: int,
    delay: float,
    allowed_exts: set[str],
    client: httpx.Client,
) -> list[str]:
    """Lightweight BFS crawl — fast but misses JS-rendered images."""
    visited_pages: set[str] = set()
    found_images: list[str] = []
    seen_images: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(normalise_url(root), 0)])

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Crawling (static)[/] {task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(root, total=None)

        while queue:
            url, depth = queue.popleft()
            norm = normalise_url(url)
            if norm in visited_pages:
                continue
            visited_pages.add(norm)
            progress.update(task, description=f"[dim]{url[:70]}[/]")

            try:
                resp = client.get(url, follow_redirects=True, timeout=10)
                resp.raise_for_status()
                html = resp.text
            except Exception as exc:
                console.log(f"[yellow]Skip[/] {url} — {exc}")
                continue

            for img_url in extract_images_from_html(html, url, allowed_exts):
                norm_img = normalise_url(img_url)
                if norm_img not in seen_images:
                    seen_images.add(norm_img)
                    found_images.append(img_url)

            if depth < max_depth:
                for link in extract_links_from_html(html, url, root):
                    if normalise_url(link) not in visited_pages:
                        queue.append((link, depth + 1))

            time.sleep(delay)

    console.log(
        f"[green]Crawled[/] {len(visited_pages)} pages, found [bold]{len(found_images)}[/] images"
    )
    return found_images


# ── downloader ────────────────────────────────────────────────────────────────


def download_one(
    img_url: str, out_dir: Path, min_bytes: int, client: httpx.Client
) -> tuple[str, str]:
    filename = sanitise_filename(img_url)
    dest = out_dir / filename
    if dest.exists():
        return img_url, "skip"
    try:
        resp = client.get(img_url, follow_redirects=True, timeout=15)
        resp.raise_for_status()
        data = resp.content
        if len(data) < min_bytes:
            return img_url, f"too_small ({len(data) // 1024}KB)"
        dest.write_bytes(data)
        return img_url, f"ok ({len(data) // 1024}KB)"
    except Exception as exc:
        return img_url, f"error: {exc}"


def download_all(
    image_urls: list[str],
    out_dir: Path,
    min_kb: int,
    workers: int,
    client: httpx.Client,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    min_bytes = min_kb * 1024
    stats: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Downloading"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("images", total=len(image_urls))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_one, url, out_dir, min_bytes, client): url
                for url in image_urls
            }
            for future in as_completed(futures):
                _, status = future.result()
                key = status.split(" ")[0].replace(":", "")
                stats[key] = stats.get(key, 0) + 1
                progress.advance(task)

    return stats


# ── Instagram ────────────────────────────────────────────────────────────────


def extract_cookies(cookie_file: Path) -> None:
    """Open browser, wait for manual login, save cookies to JSON."""
    from playwright.sync_api import sync_playwright

    console.rule("[bold cyan]Cookie Extractor[/]")
    console.print("  1. A browser will open at instagram.com/accounts/login")
    console.print("  2. Log in with your account")
    console.print("  3. Once your feed loads, come back here and press [bold]Enter[/]")
    input("\nPress Enter when you are logged in...")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = ctx.new_page()
        page.goto(
            "https://www.instagram.com/accounts/login/", wait_until="networkidle", timeout=30_000
        )
        console.print(
            "[yellow]Log in to Instagram in the browser window, then press Enter here.[/]"
        )
        input("Press Enter after logging in...")
        cookies = ctx.cookies()
        browser.close()

    import json

    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text(json.dumps(cookies, indent=2))
    # Verify sessionid was captured
    sessionid = next((c["value"] for c in cookies if c["name"] == "sessionid"), "")
    if sessionid:
        console.print(f"[green]✓ Saved {len(cookies)} cookies[/] (sessionid ✓) → {cookie_file}")
    else:
        console.print(
            "[red]sessionid not found — make sure you completed login before pressing Enter[/]"
        )


def cmd_instagram(args: argparse.Namespace) -> None:
    """Download images from an Instagram profile using Playwright."""
    import json as _json

    from playwright.sync_api import sync_playwright

    # ── extract-cookies mode ──────────────────────────────────────────────────
    if args.extract_cookies:
        extract_cookies(Path(args.cookie_file))
        return

    if not args.target and not getattr(args, "url", []):
        console.print("[red]Provide --target <username> or --url <post_url> ...[/]")
        return

    direct_urls: list[str] = getattr(args, "url", [])
    label = args.target or "direct"
    out_dir = Path(args.out) if args.out else Path("datasets") / "instagram" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    limit = args.posts
    offset = args.offset
    save_fmt = args.format  # "jpg" | "png" | "webp"
    profile_url = (
        f"https://www.instagram.com/{args.target}/" if args.target else "https://www.instagram.com/"
    )

    console.rule(f"[bold magenta]Instagram → {label}[/]")
    if direct_urls:
        console.print(
            f"  mode=direct-url  urls={len(direct_urls)}  fmt={save_fmt}  output → [cyan]{out_dir}[/]\n"
        )
    else:
        page_num = offset // limit + 1 if limit else 1
        console.print(
            f"  posts={limit or 'all'}  offset={offset}  page≈{page_num}  fmt={save_fmt}  output → [cyan]{out_dir}[/]\n"
        )

    IG_CDN = ("cdninstagram.com", "fbcdn.net")
    # Only skip known avatar/tiny thumbnail size suffixes — e35 is full-res, do NOT skip it
    SKIP_PATTERNS = ("s150x150", "s320x320", "s480x480", "s640x640", "profile_pic")

    saved_files: list[Path] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )

        # ── load cookies: JSON file takes priority over instaloader pickle ────
        cookie_file = Path(args.cookie_file)
        session_file = Path(args.session_file.replace("%(username)s", args.username))

        if cookie_file.exists():
            try:
                cookies = _json.loads(cookie_file.read_text())
                sessionid = next((c["value"] for c in cookies if c["name"] == "sessionid"), "")
                if not sessionid:
                    console.print(
                        "[red]sessionid empty in cookie file.[/] "
                        "Re-run: [bold]uv run scrape_images.py ig --extract-cookies[/]"
                    )
                    browser.close()
                    return
                ctx.add_cookies(cookies)
                console.log(
                    f"[green]Loaded {len(cookies)} cookies[/] from {cookie_file} (sessionid ✓)"
                )
            except Exception as exc:
                console.log(f"[yellow]Cookie file error:[/] {exc}")

        elif session_file.exists() and args.username:
            # fallback: try instaloader pickle
            try:
                import pickle

                with open(session_file, "rb") as f:
                    d = pickle.load(f)
                cookie_dict = d if isinstance(d, dict) else {c.name: c.value for c in d.cookies}
                sessionid = cookie_dict.get("sessionid", "")
                if not sessionid:
                    console.print(
                        "[red]sessionid empty in instaloader session.[/]\n"
                        "Run: [bold]uv run scrape_images.py ig --extract-cookies[/]"
                    )
                    browser.close()
                    return
                ig_cookies = [
                    {
                        "name": k,
                        "value": str(v),
                        "domain": ".instagram.com",
                        "path": "/",
                        "secure": True,
                        "httpOnly": k in ("sessionid", "csrftoken"),
                        "sameSite": "None",
                    }
                    for k, v in cookie_dict.items()
                    if v
                ]
                ctx.add_cookies(ig_cookies)
                console.log(f"[green]Loaded {len(ig_cookies)} cookies[/] from instaloader session")
            except Exception as exc:
                console.log(f"[yellow]Session load error:[/] {exc}")
        else:
            console.print(
                "[red]No cookies found.[/] Run first:\n"
                "  [bold]uv run scrape_images.py ig --extract-cookies[/]"
            )
            browser.close()
            return

        page = ctx.new_page()

        # ── intercept CDN image responses and save bytes while browser is live ─
        # CDN URLs are session-signed — they 403 if fetched externally.
        # We save bytes directly in the callback (sync_playwright makes body() safe).
        # `capturing` flag lets us ignore thumbnail noise during Phase 1 grid scroll.
        capturing = False
        intercepted_urls: set[str] = set()  # track seen to avoid duplicates

        def on_response(response: object) -> None:
            if not capturing:
                return
            url: str = response.url  # type: ignore[attr-defined]
            if not any(cdn in url for cdn in IG_CDN):
                return
            if any(pat in url for pat in SKIP_PATTERNS):
                return
            ct: str = response.headers.get("content-type", "")  # type: ignore[attr-defined]
            if "image" not in ct:
                return
            base = url.split("?")[0]
            if base in intercepted_urls:
                return
            intercepted_urls.add(base)
            try:
                data: bytes = response.body()  # type: ignore[attr-defined]
            except Exception:
                return
            if len(data) < 20 * 1024:  # skip < 20 KB (tiny thumbnails)
                return
            # Only keep JPEG/PNG/WEBP — filter out tiny UI assets under 50 KB
            if len(data) < 50 * 1024 and "png" in ct:
                return
            # derive filename from URL shortcode segment; strip any trailing extension
            sc = re.search(r"/([A-Za-z0-9_-]{11})/", base)
            raw_tag = sc.group(1) if sc else sanitise_filename(base)[-40:]
            # strip .jpg / .png / .webp suffix already present in the tag
            tag = re.sub(r"\.(jpg|jpeg|png|webp)$", "", raw_tag, flags=re.IGNORECASE)
            # number from offset so files across pages don't overwrite each other
            file_index = offset + len(saved_files) + 1
            fname = f"{file_index:04d}_{tag}.{save_fmt}"
            dest = out_dir / fname
            out_dir.mkdir(parents=True, exist_ok=True)
            # Re-encode to requested format if different from native
            native_fmt = "jpg" if "jpeg" in ct else ct.split("/")[-1].split(";")[0]
            if save_fmt != native_fmt and save_fmt in ("png", "webp", "jpg"):
                try:
                    pil_fmt = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}[save_fmt]
                    img = Image.open(io.BytesIO(data))
                    buf = io.BytesIO()
                    img.save(buf, format=pil_fmt)
                    dest.write_bytes(buf.getvalue())
                except Exception:
                    dest.write_bytes(data)  # fallback: save raw bytes
            else:
                dest.write_bytes(data)
            saved_files.append(dest)

        page.on("response", on_response)

        # ── navigate to profile ───────────────────────────────────────────────
        # Go to homepage first so Instagram recognises the session cookies,
        # then navigate to the target profile — avoids the age gate on cold load.
        console.log("[cyan]Warming up session on instagram.com...[/]")
        try:
            page.goto("https://www.instagram.com/", wait_until="domcontentloaded", timeout=20_000)
            page.wait_for_timeout(2000)
        except Exception:
            pass

        # dismiss any cookie/consent popup on the homepage
        for selector in [
            "button._a9--._ap36._a9_1",
            "button:has-text('Allow')",
            "button:has-text('Accept')",
            "button:has-text('Allow all cookies')",
        ]:
            try:
                btn = page.query_selector(selector)
                if btn and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(500)
            except Exception:
                pass

        # verify we're actually logged in — check page text and URL
        page_text_home = page.inner_text("body") if page.query_selector("body") else ""
        logged_in = (
            page.query_selector("svg[aria-label='Home']") is not None
            or page.query_selector("a[href='/direct/inbox/']") is not None
            or "accounts/login" not in page.url
            or "Log in" not in page_text_home[:500]
        )
        if logged_in:
            console.log("[green]Session active[/] — proceeding to profile")
        else:
            console.log("[yellow]Login status unclear — proceeding anyway[/]")

        console.log(f"[cyan]Navigating to[/] {profile_url}")
        try:
            page.goto(profile_url, wait_until="networkidle", timeout=25_000)
        except Exception as exc:
            console.log(f"[yellow]Navigation warning:[/] {exc}")
        page.wait_for_timeout(2500)

        # ── handle age gate & popups ───────────────────────────────────────────
        page_text = page.inner_text("body") if page.query_selector("body") else ""
        if "21 years" in page_text or (
            "age" in page_text.lower() and "restricted" in page_text.lower()
        ):
            # One more reload — sometimes cookies need a round-trip to activate
            console.log("[yellow]Age gate detected — retrying after reload...[/]")
            try:
                page.reload(wait_until="networkidle", timeout=20_000)
                page.wait_for_timeout(2000)
            except Exception:
                pass

        # dismiss login/consent popups on the profile page
        for selector in [
            "button[class*='HoLwm']",
            "div[role='dialog'] button",
            "button:has-text('Not Now')",
            "button:has-text('Accept')",
        ]:
            try:
                btn = page.query_selector(selector)
                if btn and btn.is_visible():
                    btn.click()
                    page.wait_for_timeout(600)
            except Exception:
                pass

        # final check — abort with a clear message if still gated
        page_text = page.inner_text("body") if page.query_selector("body") else ""
        if "21 years" in page_text or (
            "restricted" in page_text.lower() and "log in" in page_text.lower()
        ):
            console.print(
                "[red]Age gate still active.[/]\n\n"
                "Your [bold]a_highway[/] account needs age verification:\n"
                "  1. Open Instagram app on your phone\n"
                "  2. Settings → Account → Age verification → confirm birthdate\n"
                "  3. Re-login: [bold]uv run python -m instaloader --login a_highway[/]\n"
                "  4. Retry this command"
            )
            browser.close()
            return

        # ── Phase 1: scroll profile grid and collect post / reel links ───────
        if direct_urls:
            # Direct URL mode: skip profile grid entirely
            target_links = direct_urls
            console.log(f"[cyan]Direct URL mode[/] — {len(target_links)} post(s) supplied")
        else:
            console.log("[cyan]Phase 1: scrolling grid to collect post links...[/]")
            collected_links: list[str] = []
            seen_links: set[str] = set()
            stall = 0
            want = (offset + limit) if limit else 9999

            while len(collected_links) < want:
                hrefs: list[str] = page.evaluate("""
                    () => Array.from(
                        document.querySelectorAll('a[href*="/p/"], a[href*="/reel/"]')
                    ).map(a => a.href)
                """)
                before = len(collected_links)
                for href in hrefs:
                    clean = href.split("?")[0].rstrip("/")
                    if clean not in seen_links:
                        seen_links.add(clean)
                        collected_links.append(clean)

                stall = 0 if len(collected_links) > before else stall + 1
                if stall >= 10:
                    break

                page.evaluate("window.scrollBy(0, window.innerHeight * 3)")
                page.wait_for_timeout(int(args.delay * 1000) + 1200)

            # Apply offset + limit window
            end = (offset + limit) if limit else len(collected_links)
            target_links = collected_links[offset:end]
            console.log(
                f"[green]Found {len(collected_links)} total links[/], "
                f"using [{offset}:{end}] → [bold]{len(target_links)}[/] posts"
            )

        # ── Phase 2: open each post so the full-res CDN URL fires on_response ─
        # Enable byte capture now (disabled during Phase 1 to skip thumbnails)
        capturing = True  # type: ignore[assignment]  # nonlocal-style flag for closure
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Opening posts[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[dim]{task.completed}/" + str(len(target_links)) + " posts[/]"),
            TextColumn("  [green]{task.fields[imgs]} imgs saved[/]"),
            console=console,
        ) as progress:
            task = progress.add_task("posts", total=len(target_links), imgs=0)
            for post_url in target_links:
                try:
                    page.goto(post_url, wait_until="domcontentloaded", timeout=15_000)
                    page.wait_for_timeout(1500)  # let first image CDN request fire

                    # ── carousel: click → Next until the button disappears ──
                    # Instagram uses aria-label="Next" on the right-arrow button
                    for _ in range(20):  # max 20 slides per post
                        next_btn = page.query_selector(
                            'button[aria-label="Next"], '
                            'button[aria-label="next"], '
                            'svg[aria-label="Next"]'
                        )
                        if not next_btn:
                            break
                        try:
                            next_btn.click()
                            page.wait_for_timeout(800)  # wait for next slide CDN request
                        except Exception:
                            break
                except Exception:
                    pass
                progress.advance(task)
                progress.update(task, imgs=len(saved_files))
                time.sleep(args.delay * 0.3)  # polite throttle

        browser.close()

    # ── summary ───────────────────────────────────────────────────────────────
    console.rule("[bold green]Done[/]")
    t = Table(show_header=False, box=None)
    t.add_row("[green]✓ images saved[/]", str(len(saved_files)))
    t.add_row("[cyan]output dir[/]", str(out_dir))
    console.print(t)
    if saved_files:
        console.print(f"\nFirst file: [dim]{saved_files[0].name}[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image scraper for LoRA training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── instagram sub-command ─────────────────────────────────────────────────
    ig = sub.add_parser("instagram", aliases=["ig"], help="Download via Playwright browser")
    ig.add_argument("--target", default="", help="Username to scrape")
    ig.add_argument(
        "--url",
        nargs="+",
        default=[],
        metavar="URL",
        help="One or more direct post/reel URLs to download (skips profile grid)",
    )
    ig.add_argument("--posts", type=int, default=30, help="Number of posts per run (default: 30)")
    ig.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many posts before downloading (for pagination)",
    )
    ig.add_argument("--out", default="", help="Output directory")
    ig.add_argument("--delay", type=float, default=1.5, help="Scroll delay in seconds")
    ig.add_argument(
        "--format",
        choices=["jpg", "png", "webp"],
        default="jpg",
        help="Output image format (default: jpg). png/webp re-encodes via Pillow.",
    )
    ig.add_argument("--username", default="", help="Instagram username (for session file lookup)")
    ig.add_argument(
        "--session-file",
        default="~/.config/instaloader/session-%(username)s",
        help="Instaloader session file (or use --extract-cookies)",
    )
    ig.add_argument(
        "--cookie-file",
        default="ig_cookies.json",
        help="Browser cookie JSON file (default: ig_cookies.json)",
    )
    ig.add_argument(
        "--extract-cookies",
        action="store_true",
        help="Open browser, log in manually, save cookies — run once before scraping",
    )

    # ── site sub-command ──────────────────────────────────────────────────────
    site = sub.add_parser("site", help="Crawl any website (Playwright or static)")
    site.add_argument("--url", required=True, help="Root URL to crawl")
    site.add_argument("--out", default="", help="Output directory (default: datasets/<host>)")
    site.add_argument("--depth", type=int, default=2, help="Max crawl depth")
    site.add_argument("--min-size", type=int, default=20, help="Min image size in KB")
    site.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    site.add_argument("--delay", type=float, default=0.5, help="Delay between requests (s)")
    site.add_argument("--wait", type=int, default=2000, help="Extra JS wait after page load (ms)")
    site.add_argument("--no-scroll", action="store_true", help="Disable auto-scroll")
    site.add_argument("--exts", default="jpg,jpeg,png,webp", help="Allowed extensions")
    site.add_argument("--static", action="store_true", help="Use httpx only (no browser)")
    site.add_argument("--dry-run", action="store_true", help="List images without saving")

    args = parser.parse_args()

    if args.command in ("instagram", "ig"):
        cmd_instagram(args)
        return

    # ── site command ──────────────────────────────────────────────────────────
    root_url = args.url.rstrip("/")
    host = urlparse(root_url).netloc.replace("www.", "")
    out_dir = Path(args.out) if args.out else Path("datasets") / host
    allowed_exts = {e.strip().lower() for e in args.exts.split(",")}

    mode = "[yellow]static/httpx[/]" if args.static else "[cyan]Playwright/Chromium[/]"
    console.rule(f"[bold magenta]Scraping {root_url}[/]")
    console.print(
        f"  mode={mode}  depth={args.depth}  min-size={args.min_size}KB  workers={args.workers}"
    )
    console.print(f"  output → [cyan]{out_dir}[/]\n")

    with httpx.Client(headers=HEADERS) as client:
        if args.static:
            image_urls = crawl_static(root_url, args.depth, args.delay, allowed_exts, client)
        else:
            image_urls = crawl_playwright(
                root_url, args.depth, args.delay, args.wait, not args.no_scroll, allowed_exts
            )

        if not image_urls:
            console.print("[yellow]No images found.[/]")
            return

        if args.dry_run:
            table = Table("#", "URL", show_lines=False)
            for i, url in enumerate(image_urls, 1):
                table.add_row(str(i), url)
            console.print(table)
            console.print(f"\n[bold]{len(image_urls)}[/] images found (dry-run, not saved)")
            return

        stats = download_all(image_urls, out_dir, args.min_size, args.workers, client)

    console.rule("[bold green]Done[/]")
    table = Table(show_header=False, box=None)
    table.add_row("[green]✓ saved[/]", str(stats.get("ok", 0)))
    table.add_row("[dim]skipped (exists)[/]", str(stats.get("skip", 0)))
    table.add_row("[yellow]too small[/]", str(stats.get("too_small", 0)))
    table.add_row("[red]errors[/]", str(stats.get("error", 0)))
    console.print(table)
    console.print(f"\nImages saved to [cyan]{out_dir}/[/]")


if __name__ == "__main__":
    main()
