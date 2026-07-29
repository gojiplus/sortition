"""A single self-contained HTML page summarising a window of routing traffic.

No server, no CDN, no external fonts: one file that opens from ``file://`` and
survives being emailed. Charts are inline SVG built here rather than by a
plotting library, which is what keeps the output dependency-free.

Form follows the job, in the order the questions get asked:

- *Is any of this believable?* -- a status callout, first, when it is not.
- *What happened?* -- stat tiles. A single headline number is not a chart.
- *Where did traffic go, and when?* -- composition over time, stacked area.
- *What did each arm cost and return?* -- magnitude by identity, horizontal bars.
- *Would something else have been better?* -- an estimate with uncertainty, so a
  dot with an interval, never a bare bar.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sortition.reporting import Report

if TYPE_CHECKING:
    import polars as pl

# Categorical slots in fixed order, never cycled. Both modes are selected steps
# of the same hues rather than an automatic flip. Validated with the skill's
# checker on the adjacent pairlist: worst CVD dE 9.1 light / 8.4 dark.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300")
MAX_SERIES = len(SERIES_LIGHT)

_CSS = """
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --grid: #dedcd6;
  --good: #0ca30c;
  --critical: #d03b3b;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --s4: #eda100; --s5: #e87ba4; --s6: #008300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #262624;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --grid: #3a3a37;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --s4: #c98500; --s5: #d55181; --s6: #008300;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #262624;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --grid: #3a3a37;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --s4: #c98500; --s5: #d55181; --s6: #008300;
}
.viz-root {
  background: var(--surface-1); color: var(--text-primary);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
  padding: 2rem 1.25rem; margin: 0 auto; max-width: 62rem;
}
.viz-root h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
.viz-root h2 { font-size: 1rem; margin: 2.25rem 0 .75rem; font-weight: 600; }
.viz-root .sub { color: var(--text-secondary); font-size: .875rem; margin: 0; }
.tiles { display: flex; flex-wrap: wrap; gap: .75rem; margin-top: 1.5rem; }
.tile {
  background: var(--surface-2); border-radius: 10px; padding: .8rem 1rem;
  min-width: 8.5rem; flex: 1 1 8.5rem;
}
.tile .k { color: var(--text-secondary); font-size: .78rem; }
.tile .v { font-size: 1.5rem; font-weight: 650; font-variant-numeric: tabular-nums; }
.alert {
  border-left: 4px solid var(--critical); background: var(--surface-2);
  border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; margin-top: 1.25rem;
}
.alert h2 { margin: 0 0 .4rem; color: var(--critical); font-size: .95rem; }
.alert ul { margin: 0; padding-left: 1.1rem; color: var(--text-secondary); }
.legend { display: flex; flex-wrap: wrap; gap: .9rem; margin: .5rem 0 .25rem; }
.legend span { display: flex; align-items: center; gap: .4rem;
  color: var(--text-secondary); font-size: .82rem; }
.swatch { width: 11px; height: 11px; border-radius: 3px; }
.viz-root svg { display: block; max-width: 100%; overflow: visible; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .85rem;
  font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .4rem .6rem;
  border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
details { margin-top: .75rem; }
summary { cursor: pointer; color: var(--text-secondary); font-size: .85rem; }
.foot { color: var(--text-secondary); font-size: .8rem; margin-top: 2.5rem; }
[data-tip] { cursor: default; }
"""

# A crosshair-free hover: every mark carries its own <title>, which browsers
# render natively. It needs no script, survives having JavaScript disabled, and
# is read out by screen readers -- which a custom div tooltip is not.
_NOTE = (
    "Hover any mark for its exact value. Every chart has a table view below it, "
    "so identity is never carried by colour alone."
)


@dataclass(frozen=True)
class Series:
    """One named band of values over the shared x positions."""

    name: str
    values: list[float]


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _tile(label: str, value: str) -> str:
    return (
        f'<div class="tile"><div class="k">{_esc(label)}</div>'
        f'<div class="v">{_esc(value)}</div></div>'
    )


def _legend(names: list[str]) -> str:
    items = "".join(
        f'<span><i class="swatch" style="background:var(--s{i + 1})"></i>'
        f"{_esc(n)}</span>"
        for i, n in enumerate(names)
    )
    return f'<div class="legend">{items}</div>'


def _table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        f"<details><summary>{_esc(caption)}</summary>"
        f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></details>"
    )


def _stacked_area(labels: list[str], series: list[Series]) -> str:
    """Composition over time: what share of traffic each arm took.

    Stacked because the parts sum to a meaningful whole (all traffic), which is
    the one case stacking is honest.

    Args:
        labels: X tick labels, one per position.
        series: One band per arm, values summing to 1 per position.

    Returns:
        Inline SVG.
    """
    width, height, pad_l, pad_b, pad_t = 720, 240, 44, 26, 8
    n = len(labels)
    if n < 2 or not series:
        return '<p class="sub">Not enough time buckets to plot a trend.</p>'

    plot_w, plot_h = width - pad_l - 8, height - pad_b - pad_t

    def x(i: int) -> float:
        return pad_l + plot_w * i / (n - 1)

    def y(v: float) -> float:
        return pad_t + plot_h * (1.0 - v)

    parts = [
        f'<line x1="{pad_l}" y1="{y(f):.1f}" x2="{width - 8}" y2="{y(f):.1f}" '
        f'stroke="var(--grid)" stroke-width="1"/>'
        f'<text x="{pad_l - 8}" y="{y(f) + 4:.1f}" text-anchor="end" '
        f'font-size="11" fill="var(--text-secondary)">{int(f * 100)}%</text>'
        for f in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]

    baseline = [0.0] * n
    for slot, band in enumerate(series):
        top = [baseline[i] + band.values[i] for i in range(n)]
        upper = " ".join(f"{x(i):.1f},{y(top[i]):.1f}" for i in range(n))
        lower = " ".join(f"{x(i):.1f},{y(baseline[i]):.1f}" for i in reversed(range(n)))
        share = sum(band.values) / n
        # A 2px surface-coloured stroke is the gap between stacked segments, so
        # adjacent fills never touch and the boundary stays legible.
        parts.append(
            f'<polygon points="{upper} {lower}" fill="var(--s{slot + 1})" '
            f'stroke="var(--surface-1)" stroke-width="2">'
            f"<title>{_esc(band.name)}: {share:.1%} of traffic on average</title>"
            f"</polygon>"
        )
        baseline = top

    for i in (0, n - 1):
        anchor = "start" if i == 0 else "end"
        parts.append(
            f'<text x="{x(i):.1f}" y="{height - 6}" text-anchor="{anchor}" '
            f'font-size="11" fill="var(--text-secondary)">{_esc(labels[i])}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Share of traffic by arm over time">{"".join(parts)}</svg>'
    )


def _bars(rows: list[tuple[str, float]], fmt: str, slot_of: dict[str, int]) -> str:
    """Magnitude by identity: one horizontal bar per arm.

    Horizontal because arm names are text and reading them shouldn't require
    tilting your head.

    Args:
        rows: Label and value pairs.
        fmt: Format spec for the value labels.
        slot_of: Arm to colour slot, so an arm keeps its colour across charts.

    Returns:
        Inline SVG.
    """
    if not rows:
        return '<p class="sub">No data.</p>'
    width, bar_h, gap, pad_l = 720, 22, 12, 96
    height = len(rows) * (bar_h + gap)
    top = max((v for _, v in rows), default=0.0) or 1.0
    plot_w = width - pad_l - 96

    parts = []
    for i, (label, value) in enumerate(rows):
        y = i * (bar_h + gap)
        w = max(2.0, plot_w * value / top)
        slot = slot_of.get(label, i % MAX_SERIES) + 1
        parts.append(
            f'<text x="{pad_l - 10}" y="{y + bar_h * 0.72:.0f}" text-anchor="end" '
            f'font-size="12" fill="var(--text-primary)">{_esc(label)}</text>'
            # 4px rounded data-end, anchored square to the baseline.
            f'<rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" '
            f'fill="var(--s{slot})"><title>{_esc(label)}: {value:{fmt}}</title></rect>'
            # Direct label: the value sits beside its bar, so no lookup is needed.
            f'<text x="{pad_l + w + 8:.1f}" y="{y + bar_h * 0.72:.0f}" font-size="12" '
            f'fill="var(--text-secondary)">{value:{fmt}}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Value by arm">{"".join(parts)}</svg>'
    )


def _intervals(rows: list[tuple[str, float, float, float]], fmt: str) -> str:
    """An estimate with uncertainty: a dot on its interval, never a bare bar.

    A bar would imply the estimate is the quantity. It is not -- it is a point
    inside a range, and the range is the part worth reading.

    Args:
        rows: Label, point estimate, interval low and high.
        fmt: Format spec for the labels.

    Returns:
        Inline SVG.
    """
    if not rows:
        return '<p class="sub">No comparisons available.</p>'
    width, row_h, pad_l = 720, 40, 200
    height = len(rows) * row_h
    plot_w = width - pad_l - 110

    lows = [low for _, _, low, _ in rows]
    highs = [high for _, _, _, high in rows]
    lo, hi = min([*lows, 0.0]), max([*highs, 0.0])
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return pad_l + plot_w * (v - lo) / span

    parts = [
        # Zero is the reference: an interval crossing it means "no difference".
        f'<line x1="{x(0.0):.1f}" y1="0" x2="{x(0.0):.1f}" y2="{height}" '
        f'stroke="var(--grid)" stroke-width="1" stroke-dasharray="3 3"/>'
    ]
    for i, (label, value, low, high) in enumerate(rows):
        y = i * row_h + row_h / 2
        crosses = low <= 0.0 <= high
        colour = "var(--text-secondary)" if crosses else "var(--s1)"
        note = "not distinguishable from zero" if crosses else "significant"
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + 4:.0f}" text-anchor="end" font-size="12" '
            f'fill="var(--text-primary)">{_esc(label)}</text>'
            f'<line x1="{x(low):.1f}" y1="{y:.0f}" x2="{x(high):.1f}" y2="{y:.0f}" '
            f'stroke="{colour}" stroke-width="2"/>'
            # >=8px marker, ringed in the surface colour so it stays visible
            # where it overlaps its own interval line.
            f'<circle cx="{x(value):.1f}" cy="{y:.0f}" r="5" fill="{colour}" '
            f'stroke="var(--surface-1)" stroke-width="2">'
            f"<title>{_esc(label)}: {value:{fmt}} "
            f"[{low:{fmt}}, {high:{fmt}}] — {note}</title></circle>"
            f'<text x="{width - 100}" y="{y + 4:.0f}" font-size="12" '
            f'fill="var(--text-secondary)">{value:{fmt}}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Estimated difference with confidence interval">'
        f"{''.join(parts)}</svg>"
    )


def _arm_share_over_time(
    logs: pl.DataFrame, arms: list[str]
) -> tuple[list[str], list[Series]]:
    import polars as pl

    if "ts" not in logs.columns or logs.height < 20:
        return [], []
    buckets = min(24, max(4, logs.height // 200))
    frame = (
        logs.select(["ts", "chosen_arm"])
        .sort("ts")
        .with_columns((pl.int_range(pl.len()) * buckets // pl.len()).alias("_b"))
    )
    labels, series = [], {arm: [] for arm in arms}
    for bucket in range(buckets):
        rows = frame.filter(pl.col("_b") == bucket)
        if rows.is_empty():
            continue
        counts = rows.get_column("chosen_arm").value_counts()
        total = rows.height
        mapping = dict(zip(counts["chosen_arm"], counts["count"], strict=True))
        for arm in arms:
            series[arm].append(mapping.get(arm, 0) / total)
        labels.append(str(rows.get_column("ts").item(0))[:16])
    return labels, [Series(arm, series[arm]) for arm in arms]


def render(report: Report, logs: pl.DataFrame) -> str:
    """Build the dashboard page.

    Args:
        report: The assembled report, for health and comparisons.
        logs: The log window, for the per-arm and over-time charts.

    Returns:
        A complete, self-contained HTML document.
    """
    arms = sorted(report.health.arm_share)[:MAX_SERIES]
    slot_of = {arm: i for i, arm in enumerate(arms)}

    blocks: list[str] = [
        "<h1>Routing report</h1>",
        f'<p class="sub">{_esc(report.window)} &middot; '
        f"{report.generated_at:%Y-%m-%d %H:%M UTC} &middot; "
        f"{report.health.n:,} evaluable requests</p>",
    ]

    # The verdict goes above every number anyone might act on. A savings figure
    # from a blind log renders exactly like one from a good log.
    if not report.trustworthy:
        warnings = "".join(f"<li>{_esc(w)}</li>" for w in report.health.warnings)
        blocks.append(
            '<div class="alert"><h2>These numbers should not be acted on</h2>'
            f"<ul>{warnings}</ul></div>"
        )

    spend = sum(
        report.observed.get(m, 0.0) * report.health.n
        for m in ("cost_usd",)
        if m in report.observed
    )
    blocks.append(
        '<div class="tiles">'
        + _tile("requests", f"{report.health.n:,}")
        + _tile("exploration", f"{report.health.exploration_rate:.1%}")
        + _tile("leakage", f"{report.health.leakage_rate:.1%}")
        + _tile("spend", f"${spend:,.2f}")
        + _tile("healthy", "yes" if report.trustworthy else "no")
        + "</div>"
    )

    labels, series = _arm_share_over_time(logs, arms)
    if series:
        blocks += [
            "<h2>Where traffic went</h2>",
            _legend([s.name for s in series]),
            _stacked_area(labels, series),
            _table(
                ["arm", "average share"],
                [[s.name, f"{sum(s.values) / len(s.values):.1%}"] for s in series],
                "Table view",
            ),
        ]

    share_rows = [(arm, report.health.arm_share[arm]) for arm in arms]
    blocks += [
        "<h2>Share of requests by arm</h2>",
        _bars(share_rows, ".1%", slot_of),
        _table(
            ["arm", "share"], [[a, f"{v:.2%}"] for a, v in share_rows], "Table view"
        ),
    ]

    for baseline, results in report.comparisons.items():
        rows = [
            (
                f"{r.metric}: {r.b_name} vs {r.a_name}",
                r.difference,
                r.difference_interval[0],
                r.difference_interval[1],
            )
            for r in results
        ]
        blocks += [
            f"<h2>Against {_esc(baseline)}</h2>",
            '<p class="sub">Point estimate on its 95% interval. An interval '
            "crossing the dashed zero line means the difference is not "
            "distinguishable from none.</p>",
            _intervals(rows, "+.4g"),
            _table(
                ["metric", "baseline", "alternative", "difference", "95% interval"],
                [
                    [
                        r.metric,
                        f"{r.a.value:.6g}",
                        f"{r.b.value:.6g}",
                        f"{r.difference:+.6g}",
                        f"[{r.difference_interval[0]:+.4g}, "
                        f"{r.difference_interval[1]:+.4g}]",
                    ]
                    for r in results
                ],
                "Table view",
            ),
        ]

    blocks.append(f'<p class="foot">{_esc(_NOTE)}</p>')
    body = "\n".join(blocks)
    title = f"sortition — routing report ({report.window})"
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head>"
        f"<body><main class='viz-root'>{body}</main></body></html>"
    )


def write(report: Report, logs: pl.DataFrame, path: Any) -> Any:
    """Render the dashboard and write it to disk.

    Args:
        report: The assembled report.
        logs: The log window.
        path: Destination file.

    Returns:
        The path written.
    """
    from pathlib import Path

    target = Path(path)
    target.write_text(render(report, logs), encoding="utf-8")
    return target


__all__ = ["Series", "render", "write"]
