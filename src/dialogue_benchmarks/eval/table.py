from dataclasses import dataclass, field


@dataclass
class ResultsTable:
    methods: list
    metrics: list
    rows: dict  # method_name -> {metric_name: float}
    # method_name -> {metric_name: {category: float}} -- only populated for
    # metrics/items that carry item.metadata["category"]; empty dict for a
    # method/metric with no category breakdown. See eval.paper_baselines for
    # comparing this against a dataset's published baselines.
    by_category: dict = field(default_factory=dict)

    def flat_rows(self) -> tuple:
        """(columns, {method: {column: value}}) with dict-valued metrics (an
        aggregate() may return e.g. {"R@1": .., "R@5": ..}) expanded into one
        "metric.key" column per sub-score, in first-seen order."""
        columns, flat = [], {}
        for method in self.methods:
            flat[method] = {}
            for m in self.metrics:
                v = self.rows[method].get(m)
                if isinstance(v, dict):
                    for k, sub in v.items():
                        flat[method][f"{m}.{k}"] = sub
                else:
                    flat[method][m] = v
            for col in flat[method]:
                if col not in columns:
                    columns.append(col)
        return columns, flat

    def __repr__(self) -> str:
        columns, flat = self.flat_rows()
        name_w = max([len(m) for m in self.methods] + [6]) + 2
        col_w = max([len(c) for c in columns] + [6]) + 2

        def _cell(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return f"{v:>{col_w}.3f}"
            return f"{'-' if v is None else str(v):>{col_w}}"

        header = " " * name_w + "".join(f"{c:>{col_w}}" for c in columns)
        lines = [header]
        for method in self.methods:
            cells = "".join(_cell(flat[method].get(c)) for c in columns)
            lines.append(f"{method:<{name_w}}{cells}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {m: dict(self.rows[m]) for m in self.methods}
