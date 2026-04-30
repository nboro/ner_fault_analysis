# ==== Imports =====
import pandas as pd
import logging
import ast
from dash.exceptions import PreventUpdate
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff

# ==== Logging =====
logger = logging.getLogger(__name__)

# ==== Mock data (for demonstration and reproducibility) ====
# The original data ingestion has been removed because the underlying
# corpus is proprietary. For demonstration purposes this file builds a
# small mock dataset in memory that reproduces the figure shown in the
# paper. The example uses labels from the category schema defined in
# the paper (NAME, NAME_GIVEN, LOCATION_CITY, LOCATION_ADDRESS_STREET,
# LOCATION_ZIP). All identifiers (names, streets, postal codes) are
# randomly chosen and do not correspond to real individuals or
# locations.

true_column = "true_labels_nervaluate"
pred_column = "pred_labels_nervaluate"

import os
import re

# Path to the speaker-prefixed transcript file. Lines look like
# "Agent: ..." or "Caller: ...". The processing below strips those
# prefixes and concatenates the lines into a single space-separated
# string, so span offsets and rendering match the original mock setup.
MOCK_TRANSCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "mock_transcript.txt"
)


def _load_mock_text(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()
    cleaned = []
    speaker_re = re.compile(r"^\s*(Agent|Caller)\s*:\s*", re.IGNORECASE)
    for line in raw_lines:
        if not line.strip():
            continue
        # Replace the "Agent:"/"Caller:" prefix with a plain lowercase
        # speaker word so the output reads like a continuous transcript
        # (e.g. "agent good afternoon ... caller yes good afternoon ...").
        cleaned.append(
            speaker_re.sub(lambda m: m.group(1).lower() + " ", line).strip()
        )
    return " ".join(cleaned)


MOCK_TEXT = _load_mock_text(MOCK_TRANSCRIPT_PATH)


def _locate(text, needle, start=0):
    idx = text.index(needle, start)
    return idx, idx + len(needle)


# Card 1 entity: full name span
_c1_start, _c1_end = _locate(MOCK_TEXT, "Jon Dough")
# First occurrence of the city (next to the name)
_city1_start, _city1_end = _locate(MOCK_TEXT, "Brussels")

# Card 2 entity: postal code
_c2_start, _c2_end = _locate(MOCK_TEXT, "1111 XA")
# Street name
_street_start, _street_end = _locate(MOCK_TEXT, "84 Maple Street")
# Second occurrence of the city (in the address context)
_second_city_start, _second_city_end = _locate(MOCK_TEXT, "Brussels", _city1_end)

# Predicted spans (as a model under study would produce):
#   Card 1: boundaries match ground truth, label differs (NAME_GIVEN vs NAME).
#           This prediction is correct under Exact, incorrect under Strict.
#   Card 2: boundaries and label both match ground truth.
#           This prediction is correct under both policies.
pred_labels = [
    {"text": "Jon Dough", "label": "NAME_GIVEN",
     "start": _c1_start, "end": _c1_end},
    {"text": "1111 XA", "label": "LOCATION_ZIP",
     "start": _c2_start, "end": _c2_end},
]

# Ground truth annotations
true_labels = [
    {"text": "Jon Dough", "label": "NAME",
     "start": _c1_start, "end": _c1_end},
    {"text": "Brussels", "label": "LOCATION_CITY",
     "start": _city1_start, "end": _city1_end},
    {"text": "84 Maple Street", "label": "LOCATION_ADDRESS_STREET",
     "start": _street_start, "end": _street_end},
    {"text": "1111 XA", "label": "LOCATION_ZIP",
     "start": _c2_start, "end": _c2_end},
    {"text": "Brussels", "label": "LOCATION_CITY",
     "start": _second_city_start, "end": _second_city_end},
]

# DataFrames that the callbacks consume. The schema matches what the
# original data ingestion would have produced.
anonymization_spans = pd.DataFrame({
    "contact_text": [MOCK_TEXT],
    true_column: [str(true_labels)],
    pred_column: [str(pred_labels)],
})

anonymization_indices = pd.DataFrame({
    "measure": ["correct_indices", "incorrect_indices",
                "partial_indices", "missed_indices", "spurious_indices"],
    "label": [None, None, None, None, None],
    "strict":   ["[(0, 1)]",
                 "[(0, 0)]",
                 "[]",
                 "[(0, 1), (0, 2), (0, 4)]",
                 "[]"],
    "exact":    ["[(0, 0), (0, 1)]",
                 "[]",
                 "[]",
                 "[(0, 1), (0, 2), (0, 4)]",
                 "[]"],
    "partial":  ["[(0, 0), (0, 1)]",
                 "[]",
                 "[]",
                 "[(0, 1), (0, 2), (0, 4)]",
                 "[]"],
    "ent_type": ["[(0, 0), (0, 1)]",
                 "[]",
                 "[]",
                 "[(0, 1), (0, 2), (0, 4)]",
                 "[]"],
})

anonymization_metrics = pd.DataFrame([
    {"policy": "strict",   "label": None, "correct": 1, "incorrect": 1,
     "partial": 0, "missed": 3, "spurious": 0, "possible": 5, "actual": 2},
    {"policy": "exact",    "label": None, "correct": 2, "incorrect": 0,
     "partial": 0, "missed": 3, "spurious": 0, "possible": 5, "actual": 2},
    {"policy": "partial",  "label": None, "correct": 2, "incorrect": 0,
     "partial": 0, "missed": 3, "spurious": 0, "possible": 5, "actual": 2},
    {"policy": "ent_type", "label": None, "correct": 2, "incorrect": 0,
     "partial": 0, "missed": 3, "spurious": 0, "possible": 5, "actual": 2},
])

# Parse string-encoded label lists into Python lists of dicts
anonymization_spans[true_column] = anonymization_spans[true_column].apply(ast.literal_eval)
anonymization_spans[pred_column] = anonymization_spans[pred_column].apply(ast.literal_eval)

# ==== Constants & Config =====
# Policy: paper uses Strict and Exact only. Partial and Entity Type are
# nervaluate-package primitives but not relevant to the paper's evaluation.
policy_options = [
    {'label': 'Strict', 'value': 'strict'},
    {'label': 'Exact',  'value': 'exact'},
]

# Measure: four nervaluate verdict categories plus an All-categories option
# for browsing every fault type in one pass.
ALL_MEASURES_VALUE = 'all_measures'
measure_value_to_human = {
    'correct_indices':  'Correct',
    'incorrect_indices': 'Incorrect',
    'missed_indices':    'Missed',
    'spurious_indices':  'Spurious',
}
measure_options = [{'label': 'All', 'value': ALL_MEASURES_VALUE}] + [
    {'label': 'Correct',   'value': 'correct_indices'},
    {'label': 'Incorrect', 'value': 'incorrect_indices'},
    {'label': 'Missed',    'value': 'missed_indices'},
    {'label': 'Spurious',  'value': 'spurious_indices'},
]

# Label: full paper schema (Tables/entities in the manuscript). Listed
# explicitly so the dropdown always exposes the schema the paper defines,
# independent of which labels happen to appear in the mock data.
ALL_LABELS_VALUE = 'all_labels'
PAPER_LABEL_SCHEMA = [
    'NAME',
    'NAME_GIVEN',
    'NAME_FAMILY',
    'DOB',
    'LOCATION',
    'LOCATION_ADDRESS_STREET',
    'LOCATION_ZIP',
    'LOCATION_CITY',
    'PHONE_NUMBER',
    'EMAIL_ADDRESS',
    'BANK_ACCOUNT',
]
label_options = [{'label': 'All labels', 'value': ALL_LABELS_VALUE}] + [
    {'label': _l, 'value': _l} for _l in PAPER_LABEL_SCHEMA
]

# ==== Dash app setup ====
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.external_stylesheets = [dbc.themes.BOOTSTRAP]

# ==== Layout ====
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("Fault Analysis Dashboard", className="mb-2 mt-4"),
            html.Hr()
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label("Model"),
            dcc.Dropdown(
                id='dropdown-model',
                options=[
                    {'label': 'BERT-P', 'value': 'bert_p'},
                    {'label': 'RB (WIP)', 'value': 'rb'}
                ],
                value='bert_p',
                placeholder='Select a model'
            ),
            html.Hr()
        ], md=4)
    ]),

    # Model/Policy/Label dropdowns: SHARED (always visible)
    dbc.Row([
        dbc.Col([
            dbc.Label("Policy"),
            dcc.Dropdown(
                id='dropdown-policy',
                options=policy_options,
                value=policy_options[0]['value'],
                clearable=False
            )
        ], md=4, className="mb-3"),

        dbc.Col([
            dbc.Label("Label"),
            dcc.Dropdown(
                id='dropdown-label',
                options=label_options,
                value=ALL_LABELS_VALUE,
                clearable=False
            )
        ], md=4, className="mb-3"),
    ]),

    # Tabs
    dbc.Tabs([
        dbc.Tab(label="Span Viewer", tab_id="span-viewer"),
        dbc.Tab(label="Evaluation Metrics", tab_id="eval-metrics")
    ], id="main-tabs", active_tab="span-viewer", className="mb-3"),

    html.Div(id="tab-content"),

    dcc.Store(id='measure-store', data=ALL_MEASURES_VALUE)
], fluid=True)


# ==== Helper functions ====
# ==== Verdict colors (one channel: per-span verdict) =====
# Same palette as the failure-modes diagram in the paper, so the legend
# is consistent across figure and tool.
VERDICT_COLORS = {
    "correct":   "#d4edda",
    "incorrect": "#ffe5b4",
    "spurious":  "#e2d5f0",
    "missed":    "#f5c6cb",
}

# Ground-truth highlight color. Ground-truth spans use a single color
# regardless of verdict; only predicted spans use the per-verdict palette.
GROUND_TRUTH_COLOR = "#d1ecf1"


def _verdict_for_predicted_span(pred_span, gt_spans, policy):
    """Determine the verdict for a predicted span under the given policy.

    Returns one of: 'correct', 'incorrect', 'spurious'.
    """
    p_start, p_end, p_label = pred_span['start'], pred_span['end'], pred_span['label']
    for gt in gt_spans:
        same_boundary = (gt['start'] == p_start and gt['end'] == p_end)
        same_label = (gt['label'] == p_label)
        if policy == 'strict':
            if same_boundary and same_label:
                return 'correct'
        elif policy == 'exact':
            if same_boundary:
                return 'correct'
        elif policy == 'ent_type':
            overlaps = (gt['start'] < p_end and gt['end'] > p_start)
            if overlaps and same_label:
                return 'correct'
        elif policy == 'partial':
            overlaps = (gt['start'] < p_end and gt['end'] > p_start)
            if overlaps:
                return 'correct'
    # Boundary-overlapping ground-truth span exists but did not match under
    # the active policy criteria -> incorrect. Otherwise -> spurious.
    for gt in gt_spans:
        if gt['start'] < p_end and gt['end'] > p_start:
            return 'incorrect'
    return 'spurious'


def _verdict_for_gt_span(gt_span, pred_spans, policy):
    """Determine the verdict for a ground-truth span under the given policy.

    Returns one of: 'correct', 'incorrect', 'missed'.
    """
    g_start, g_end, g_label = gt_span['start'], gt_span['end'], gt_span['label']
    for pred in pred_spans:
        same_boundary = (pred['start'] == g_start and pred['end'] == g_end)
        same_label = (pred['label'] == g_label)
        if policy == 'strict':
            if same_boundary and same_label:
                return 'correct'
        elif policy == 'exact':
            if same_boundary:
                return 'correct'
        elif policy == 'ent_type':
            overlaps = (pred['start'] < g_end and pred['end'] > g_start)
            if overlaps and same_label:
                return 'correct'
        elif policy == 'partial':
            overlaps = (pred['start'] < g_end and pred['end'] > g_start)
            if overlaps:
                return 'correct'
    for pred in pred_spans:
        if pred['start'] < g_end and pred['end'] > g_start:
            return 'incorrect'
    return 'missed'


def render_context(text, context_start, context_end, spans_with_verdict, include_label=True):
    """Render a text slice with inline highlights.

    spans_with_verdict: list of dicts with keys 'start', 'end', 'label', 'verdict'.
    The highlight color comes from the verdict (correct / incorrect / spurious / missed).
    """
    spans = [s for s in spans_with_verdict
             if s['end'] > context_start and s['start'] < context_end]
    spans = sorted(spans, key=lambda s: s['start'])
    output = []
    cursor = context_start
    for span in spans:
        span_start = max(span['start'], context_start)
        span_end = min(span['end'], context_end)
        if span_start > cursor:
            output.append(html.Span(text[cursor:span_start]))
        label = span['label']
        verdict = span['verdict']
        if verdict == "_ground_truth":
            color = GROUND_TRUTH_COLOR
        else:
            color = VERDICT_COLORS.get(verdict, "#e9ecef")
        output.append(
            html.Span([
                text[span_start:span_end],
                html.Span(f" {label}", style={
                    "fontSize": "0.7em",
                    "marginLeft": "6px",
                    "color": "#495057",
                    "fontWeight": "600"
                }) if include_label else ""
            ], style={
                "backgroundColor": color,
                "borderRadius": "0.4em",
                "padding": "0.2em 0.5em",
                "margin": "0 0.2em",
                "display": "inline",
                "fontWeight": "500"
            })
        )
        cursor = span_end
    if cursor < context_end:
        output.append(html.Span(text[cursor:context_end]))
    return html.Span(output)


def render_verdict_legend():
    """Sticky legend explaining the highlight colors.

    Predicted spans are colored by per-span verdict.
    Ground-truth spans use a single color regardless of verdict.
    """
    swatch = lambda color: html.Span(style={
        "display": "inline-block",
        "width": "14px",
        "height": "14px",
        "backgroundColor": color,
        "borderRadius": "3px",
        "marginRight": "6px",
        "verticalAlign": "middle"
    })
    item = lambda color, label: html.Span(
        [swatch(color), label],
        style={"marginRight": "20px", "fontSize": "0.9em"}
    )
    pred_section = html.Span(
        [
            html.Span("Predicted:", style={"fontWeight": "600", "marginRight": "12px"}),
            item(VERDICT_COLORS["correct"],   "correct"),
            item(VERDICT_COLORS["incorrect"], "incorrect"),
            item(VERDICT_COLORS["spurious"],  "spurious"),
            item(VERDICT_COLORS["missed"],    "missed"),
        ],
        style={"marginRight": "32px"}
    )
    gt_section = html.Span(
        [
            html.Span("Ground truth:", style={"fontWeight": "600", "marginRight": "12px"}),
            item(GROUND_TRUTH_COLOR, "annotated span"),
        ]
    )
    return html.Div(
        [pred_section, gt_section],
        style={
            "position": "sticky",
            "top": "0",
            "zIndex": "1000",
            "backgroundColor": "#ffffff",
            "padding": "10px 14px",
            "border": "1px solid #dee2e6",
            "borderRadius": "0.4em",
            "marginBottom": "16px",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
        }
    )


def render_span_card(text, all_pred_spans, all_gt_spans, policy, label_filter,
                     verdict_summary, index=None, doc_index=None):
    """Render a single transcript card.

    A card represents one transcript. The predicted and ground-truth rows
    cover the entire transcript text. Every span carries its verdict color
    (predicted row: correct/incorrect/spurious/missed; ground-truth row:
    single color regardless of verdict). When a specific label is selected,
    only spans of that label are highlighted; everything else renders as
    plain text.

    text: the full transcript text.
    all_pred_spans / all_gt_spans: every span in the transcript.
    policy: 'strict' or 'exact' — used for verdict computation.
    label_filter: ALL_LABELS_VALUE, or a specific label string.
    verdict_summary: dict mapping verdict -> count present in this transcript
        under the active filter (e.g. {'incorrect': 1, 'missed': 3}).
    """
    context_start = 0
    context_end = len(text)

    def _label_match(span_label):
        return label_filter == ALL_LABELS_VALUE or span_label == label_filter

    # ---- Predicted row spans ----
    # Predicted spans get correct/incorrect/spurious colors per their verdict.
    # Missed ground-truth spans are overlaid on the predicted row in the missed
    # color, so the predicted row carries all four verdicts.
    pred_with_verdict = []
    for p in all_pred_spans:
        if _label_match(p['label']):
            verdict = _verdict_for_predicted_span(p, all_gt_spans, policy)
            pred_with_verdict.append({**p, 'verdict': verdict})

    for g in all_gt_spans:
        if _label_match(g['label']):
            gt_verdict = _verdict_for_gt_span(g, all_pred_spans, policy)
            if gt_verdict == 'missed':
                pred_with_verdict.append({**g, 'verdict': 'missed'})

    # ---- Ground-truth row spans ----
    # Ground-truth spans use a single color regardless of verdict.
    gt_with_verdict = []
    for g in all_gt_spans:
        if _label_match(g['label']):
            gt_with_verdict.append({**g, 'verdict': '_ground_truth'})

    pred_context = render_context(text, context_start, context_end, pred_with_verdict)
    gt_context = render_context(text, context_start, context_end, gt_with_verdict)

    # ---- Header ----
    policy_human = 'Strict' if policy == 'strict' else 'Exact'
    label_human = 'All labels' if label_filter == ALL_LABELS_VALUE else label_filter
    # Verdict summary: e.g. "1 incorrect, 3 missed"
    if verdict_summary:
        verdict_text = ', '.join(f"{v} {k}"
                                 for k, v in verdict_summary.items() if v > 0)
    else:
        verdict_text = '—'

    header = dbc.Row([
        dbc.Col(html.H5(f"{index}", className="text-secondary"), width="auto"),
        dbc.Col([
            html.H5(label_human, className="card-title text-primary mb-0"),
            html.Div([
                html.Span(f"Policy: {policy_human}",
                          className="me-3 text-muted",
                          style={"fontSize": "0.85em"}),
                html.Span(f"Verdicts in transcript: {verdict_text}",
                          className="text-muted",
                          style={"fontSize": "0.85em"}),
            ], className="mt-1")
        ]),
        dbc.Col(html.Div(f"Doc {doc_index}",
                         className="text-end text-muted"), width="auto")
    ], align="center", justify="between")

    return dbc.Card([
        dbc.CardBody([
            header,
            html.Hr(className="my-2"),
            html.Div([
                html.Div("Predicted", className="fw-bold mb-1 text-secondary"),
                html.Div(pred_context, className="p-2 border rounded bg-light mb-3")
            ]),
            html.Div([
                html.Div("Ground truth", className="fw-bold mb-1 text-secondary"),
                html.Div(gt_context, className="p-2 border rounded bg-light")
            ])
        ])
    ], className="mb-4 shadow-sm border-0")


def create_metrics_summary(row, policy):
    correct = row['correct']
    incorrect = row['incorrect']
    partial = row['partial']
    missed = row['missed']
    spurious = row['spurious']
    actual = row['actual']
    possible = row['possible']

    if policy in ['strict', 'exact']:
        precision = correct / actual if actual else 0
        recall = correct / possible if possible else 0
    else:
        precision = (correct + 0.5 * partial) / actual if actual else 0
        recall = (correct + 0.5 * partial) / possible if possible else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    summary = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Metrics", className="text-muted mb-3"),
                    dbc.Row([
                        dbc.Col(html.P("Precision", className="fw-bold"), width=6),
                        dbc.Col(html.Span(f"{precision:.2%}", className="text-end"), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P("Recall", className="fw-bold"), width=6),
                        dbc.Col(html.Span(f"{recall:.2%}", className="text-end"), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P("F1 Score", className="fw-bold"), width=6),
                        dbc.Col(html.Span(f"{f1:.2%}", className="text-end"), width=6)
                    ]),
                ]),
                dbc.Col([
                    html.H5("Summary", className="card-title text-muted mb-3"),
                    dbc.Row([
                        dbc.Col(html.P("Correct", className="fw-bold"), width=6),
                        dbc.Col(dbc.Badge(f"{correct}", color="success", className="float-end mb-1"), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P("Incorrect", className="fw-bold"), width=6),
                        dbc.Col(dbc.Badge(f"{incorrect}", color="secondary", className="float-end mb-1"), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P("False negative", className="fw-bold"), width=6),
                        dbc.Col(dbc.Badge(f"{missed}", color="danger", className="float-end mb-1"), width=6)
                    ]),
                    dbc.Row([
                        dbc.Col(html.P("False positive", className="fw-bold"), width=6),
                        dbc.Col(dbc.Badge(f"{spurious}", color="warning", className="float-end mb-1 text-dark"), width=6)
                    ])
                ])
            ]),
            html.Hr(className="my-3"),
            dbc.Row([
                dbc.Col(html.P("Total Ground Truth Entities", className="fw-bold")),
                dbc.Col(html.P(f"{possible}", className="text-end"))
            ]),
            dbc.Row([
                dbc.Col(html.P("Total Predicted Entities", className="fw-bold")),
                dbc.Col(html.P(f"{actual}", className="text-end"))
            ])
        ])
    ], className="mt-3 shadow-sm border-0")
    return summary


def create_confusion_matrix_figure(row):
    tp = row['correct'] + row['incorrect'] + row['partial']
    tn = 0
    fn = row['missed']
    fp = row['spurious']

    z = [
        [fn, tn],
        [tp, fp]
    ]

    z_text = [
        [f"False Negatives<br>{fn}", f"True Negatives<br>{tn}"],
        [f"True Positives<br>{tp}", f"False Positives<br>{fp}"]
    ]

    x_labels = ["Actually Positive (1)", "Actually Negative (0)"]
    y_labels = ["Predicted Negative (0)", "Predicted Positive (1)"]

    fig = ff.create_annotated_heatmap(
        z,
        x=x_labels,
        y=y_labels,
        annotation_text=z_text,
        colorscale="Blues",
        showscale=False,
        hoverinfo="z"
    )

    fig.update_layout(
        title="Confusion Matrix",
        margin=dict(t=50),
        xaxis_title="Actual Label",
        yaxis_title="Predicted Label",
        xaxis=dict(side='top'),
    )

    return fig


# ==== Tab content rendering ====
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("dropdown-policy", "value"),
    Input("dropdown-label", "value"),
    State("measure-store", "data")
)
def render_tab_content(active_tab, policy, label, measure_value):
    if not policy:
        return html.Div("Please select a policy.", className="text-muted")
    if active_tab == "span-viewer":
        span_tab = html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Measure"),
                    dcc.Dropdown(
                        id='dropdown-measure',
                        options=measure_options,
                        value=measure_value,
                        placeholder='Select a Measure',
                        clearable=False,
                        persistence=True,
                        persistence_type='session'
                    )
                ], md=4, className="mb-3")
            ]),
            render_verdict_legend(),
            dbc.Card([
                dbc.CardHeader(
                    html.Div(id='span-summary', className='text-muted fst-italic'),
                    className='bg-white border-bottom-0'
                ),
                dbc.CardBody([
                    html.Div(id='span-cards')
                ])
            ])
        ])
        return span_tab

    elif active_tab == "eval-metrics":
        eval_tab = html.Div([
            html.Div(id='metrics-summary', className='mt-4'),
            dcc.Graph(id="confusion-matrix", className='mt-4')
        ])
        return eval_tab

    return html.Div()


# ==== Store measure value if changed ====
@app.callback(
    Output('measure-store', 'data'),
    Input('dropdown-measure', 'value'),
    prevent_initial_call=True
)
def update_measure_store(measure_value):
    if measure_value is not None:
        return measure_value
    raise PreventUpdate


# ==== Span viewer ====
@app.callback(
    Output('span-summary', 'children'),
    Output('span-cards', 'children'),
    Input('dropdown-policy', 'value'),
    Input('dropdown-measure', 'value'),
    Input('dropdown-label', 'value')
)
def update(policy, measure, label):
    if policy is None or measure is None or label is None:
        raise PreventUpdate

    label_filter = label

    def _label_match(span_label):
        return label_filter == ALL_LABELS_VALUE or span_label == label_filter

    # Resolve which verdicts qualify a transcript for inclusion.
    # ALL_MEASURES_VALUE -> any of the four verdicts qualifies.
    measure_to_verdict = {
        'correct_indices':   'correct',
        'incorrect_indices': 'incorrect',
        'missed_indices':    'missed',
        'spurious_indices':  'spurious',
    }
    if measure == ALL_MEASURES_VALUE:
        qualifying_verdicts = set(measure_to_verdict.values())
    else:
        qualifying_verdicts = {measure_to_verdict[measure]}

    # For each transcript, compute the verdict counts under the active filter.
    # A transcript qualifies if it contains at least one span whose verdict
    # is in qualifying_verdicts.
    cards = []
    qualifying_count = 0
    for docidx in range(len(anonymization_spans)):
        text = anonymization_spans['contact_text'].iloc[docidx]
        pred_spans = anonymization_spans[pred_column].iloc[docidx]
        gt_spans = anonymization_spans[true_column].iloc[docidx]

        verdict_counts = {'correct': 0, 'incorrect': 0,
                          'spurious': 0, 'missed': 0}

        for p in pred_spans:
            if not _label_match(p['label']):
                continue
            v = _verdict_for_predicted_span(p, gt_spans, policy)
            if v in verdict_counts:
                verdict_counts[v] += 1

        for g in gt_spans:
            if not _label_match(g['label']):
                continue
            v = _verdict_for_gt_span(g, pred_spans, policy)
            if v == 'missed':
                verdict_counts['missed'] += 1

        # Does this transcript have at least one qualifying verdict present?
        if not any(verdict_counts[v] > 0 for v in qualifying_verdicts):
            continue

        qualifying_count += 1
        # Trim verdict_summary shown in header to the qualifying verdicts only.
        verdict_summary = {v: verdict_counts[v] for v in qualifying_verdicts
                           if verdict_counts[v] > 0}

        cards.append(render_span_card(
            text=text,
            all_pred_spans=pred_spans,
            all_gt_spans=gt_spans,
            policy=policy,
            label_filter=label_filter,
            verdict_summary=verdict_summary,
            index=qualifying_count,
            doc_index=docidx,
        ))

    summary_text = f"{qualifying_count} transcript(s) found"
    return summary_text, cards


# ==== Evaluation metrics ====
@app.callback(
    Output('metrics-summary', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('dropdown-policy', 'value'),
    Input('dropdown-label', 'value')
)
def update_evaluation_graph(policy, label):
    if policy is None:
        raise PreventUpdate

    metrics = anonymization_metrics[anonymization_metrics['policy'] == policy]

    if label is not None:
        metrics = metrics[metrics['label'] == label]

    metrics = metrics[['correct', 'incorrect', 'partial', 'missed',
                       'spurious', 'possible', 'actual']].sum()
    metrics_summary = create_metrics_summary(metrics, policy)
    confusion_matrix_fig = create_confusion_matrix_figure(metrics)
    return metrics_summary, confusion_matrix_fig


# ==== Entry point ====
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)