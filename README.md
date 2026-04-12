uv run scrape_images.py ig --extract-cookies

uv run scrape_images.py ig --target jel___ly --posts 30 --offset 0 --format png

uv run scrape_images.py ig \
  --url https://www.instagram.com/p/ABC123/ \
  --out datasets/instagram/spectre.a.i \
  --format png