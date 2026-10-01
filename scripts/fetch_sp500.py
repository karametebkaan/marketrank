#!/usr/bin/env python3
"""Refresh data/universe/sp500.csv (ticker,name,sector) from Wikipedia's constituents table."""
import csv
import sys
import urllib.request
from html.parser import HTMLParser

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class ConstituentsParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0  # table nesting depth inside the constituents table
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.depth > 0:
                self.depth += 1
            elif dict(attrs).get("id") == "constituents":
                self.depth = 1
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_endtag(self, tag):
        if tag == "table" and self.depth > 0:
            self.depth -= 1
        elif self.depth == 1 and tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            if self.row:
                self.rows.append(self.row)
            self.row = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def main(out_path):
    req = urllib.request.Request(URL, headers={"User-Agent": "fluxscape/0.1 (universe refresh)"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    parser = ConstituentsParser()
    parser.feed(html)
    header, *body = parser.rows
    i_sym, i_name, i_sector = (header.index("Symbol"), header.index("Security"),
                               header.index("GICS Sector"))
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "name", "sector"])
        for r in body:
            w.writerow([r[i_sym], r[i_name], r[i_sector]])
    print(f"wrote {len(body)} constituents to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/universe/sp500.csv")
