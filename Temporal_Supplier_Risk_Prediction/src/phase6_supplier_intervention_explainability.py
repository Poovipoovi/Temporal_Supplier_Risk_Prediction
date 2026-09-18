from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIORITY_FILE = ROOT / 'outputs/predictions/supplier_risk_prioritization.csv'
LOCAL_EXPL_FILE = ROOT / 'outputs/explainability/local_prediction_explanations.csv'
LATEST_EXPL_FILE = ROOT / 'outputs/explainability/latest_supplier_explanations.csv'
OUTDIR = ROOT / 'outputs/interventions'
REPORTDIR = ROOT / 'outputs/reports'
OUTDIR.mkdir(parents=True, exist_ok=True)
REPORTDIR.mkdir(parents=True, exist_ok=True)

REQ = [
    'supplier_id', 'period', 'predicted_risk_N1', 'probability_high',
    'priority_score', 'priority_band', 'delivery_risk_score',
    'quality_risk_score', 'strategic_exposure_score', 'deterioration_score',
    'model_high_risk_score'
]
FUTURE = {
    'actual_risk_N1', 'target_risk_N1', 'target_risk_score_N1',
    'prediction_correct', 'target_low_boundary', 'target_high_boundary'
}


def num(row, c, default=0.0):
    x = pd.to_numeric(pd.Series([row.get(c, default)]), errors='coerce').iloc[0]
    return default if pd.isna(x) else float(x)


def fmt_pct(x):
    return f'{float(x):.1%}'


def clean_feature_name(x):
    return str(x).strip()


def find_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_explainability():
    """Load Phase 5 local explanations if present.

    The script intentionally does not guess a schema. It supports several common
    Phase 5 column names and otherwise returns an empty explanation table so that
    operational recommendations remain deterministic and are clearly labelled.
    """
    candidates = [LOCAL_EXPL_FILE, LATEST_EXPL_FILE]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return pd.DataFrame(), 'Phase 5 local explanation file not found; score-based evidence only.'

    # Prefer the full 278-row local explanation file.
    path = existing[0]
    ex = pd.read_csv(path)
    if 'supplier_id' not in ex.columns:
        raise ValueError(f'Phase 5 explanation file exists but lacks supplier_id: {path}')

    period_col = find_column(ex, ['period', 'period_N'])
    feature_col = find_column(ex, ['feature', 'feature_name', 'variable'])
    contribution_col = find_column(ex, [
        'contribution', 'high_risk_contribution', 'high_risk_contribution_value',
        'signed_contribution', 'local_contribution'
    ])
    direction_col = find_column(ex, ['direction', 'contribution_direction'])

    if feature_col is None or contribution_col is None:
        return pd.DataFrame(), (
            f'Phase 5 explanation file found at {path}, but no supported feature/contribution '
            'columns were detected; score-based evidence only.'
        )

    ex['_feature'] = ex[feature_col].map(clean_feature_name)
    ex['_contribution'] = pd.to_numeric(ex[contribution_col], errors='coerce')
    ex = ex[ex['_feature'].ne('') & ex['_contribution'].notna()].copy()
    if ex.empty:
        return pd.DataFrame(), 'Phase 5 explanation file contained no usable local contributions.'

    # Keep only the current/latest supplier period when period is available.
    if period_col:
        ex['_period'] = ex[period_col].astype(str)
    else:
        ex['_period'] = ''

    rows = []
    for supplier, g in ex.groupby('supplier_id', sort=False):
        # Prefer the latest period represented for that supplier.
        if g['_period'].ne('').any():
            latest = g['_period'].max()
            gg = g[g['_period'] == latest].copy()
        else:
            gg = g.copy()
            latest = ''

        gg['_abs'] = gg['_contribution'].abs()
        top = gg.sort_values('_abs', ascending=False).head(5)
        positive = gg[gg['_contribution'] > 0].sort_values('_contribution', ascending=False).head(3)
        negative = gg[gg['_contribution'] < 0].sort_values('_contribution', ascending=True).head(3)

        rows.append({
            'supplier_id': supplier,
            '_explanation_period': latest,
            'top_local_drivers': ' | '.join(
                f"{r['_feature']} ({r['_contribution']:+.4f})" for _, r in top.iterrows()
            ),
            'positive_local_drivers': ' | '.join(
                f"{r['_feature']} ({r['_contribution']:+.4f})" for _, r in positive.iterrows()
            ),
            'opposing_local_drivers': ' | '.join(
                f"{r['_feature']} ({r['_contribution']:+.4f})" for _, r in negative.iterrows()
            ),
            'local_explanation_available': True
        })
    return pd.DataFrame(rows), f'Phase 5 local explanations loaded from {path}. '


def operational_reason(r):
    d = num(r, 'delivery_risk_score')
    q = num(r, 'quality_risk_score')
    s = num(r, 'strategic_exposure_score')
    det = num(r, 'deterioration_score')
    m = num(r, 'model_high_risk_score')
    p = num(r, 'probability_high')
    worsening = num(r, 'is_worsening')
    risk_change = num(r, 'risk_change')

    domains = []
    evidence = []
    actions = []

    late = num(r, 'late_delivery_rate')
    avg_late = num(r, 'avg_days_late')
    short_rate = num(r, 'short_delivery_rate')
    qr = num(r, 'quality_event_rate')
    scrap = num(r, 'total_scrap_qty')
    rework = num(r, 'rework_cost_ratio')
    spend = num(r, 'spend_share')

    if d >= 60:
        domains.append('Delivery')
        evidence.append(f'Delivery risk score {d:.1f}/100; late-delivery rate {fmt_pct(late)}; average delay {avg_late:.2f} days; short-delivery rate {fmt_pct(short_rate)}.')
        actions.append('Initiate supplier delivery corrective action plan; review late orders, lead-time causes and recovery dates.')
    elif d >= 40:
        domains.append('Delivery')
        evidence.append(f'Delivery risk score {d:.1f}/100; late-delivery rate {fmt_pct(late)}; average delay {avg_late:.2f} days.')
        actions.append('Increase delivery-performance monitoring and require a short-term recovery plan.')

    if q >= 60:
        domains.append('Quality')
        evidence.append(f'Quality risk score {q:.1f}/100; quality-event rate {fmt_pct(qr)}; scrap quantity {scrap:.2f}; rework-cost ratio {rework:.2%}.')
        actions.append('Initiate supplier quality corrective action; review quality events, scrap and rework controls.')
    elif q >= 30:
        domains.append('Quality')
        evidence.append(f'Quality risk score {q:.1f}/100; quality-event rate {fmt_pct(qr)}; scrap quantity {scrap:.2f}.')
        actions.append('Increase quality surveillance and review recurring defects, scrap and rework.')

    if s >= 20:
        domains.append('Strategic exposure')
        evidence.append(f'Strategic exposure score {s:.1f}/100; spend share {spend:.2%}.')
        actions.append('Perform supply-continuity review and assess alternate-source or capacity options.')

    if det >= 70 or worsening >= 1 or risk_change > 0:
        domains.append('Deterioration')
        evidence.append(f'Deterioration score {det:.1f}/100; is_worsening={int(worsening)}; risk_change={risk_change:+.2f}.')
        actions.append('Increase review frequency and investigate the recent negative performance trend; track a dated recovery plan.')

    if m >= 50:
        domains.append('Model signal')
        evidence.append(f'Model high-risk score {m:.1f}/100; predicted High probability {p:.2%}.')
        actions.append('Place supplier on enhanced risk monitoring because the selected model assigns a material high-risk probability.')
    elif m >= 30:
        domains.append('Model signal')
        evidence.append(f'Model high-risk score {m:.1f}/100; predicted High probability {p:.2%}.')
        actions.append('Review the supplier model-risk signal alongside operational evidence during the next supplier review.')

    if not actions:
        domains = ['Routine monitoring']
        evidence = ['No operational component crossed the Phase 6 intervention thresholds.']
        actions = ['Continue routine supplier performance monitoring.']

    # Use the strongest operational domain as the primary reason. Model signal is
    # deliberately last because it should support, not replace, operational evidence.
    primary = domains[0]
    return domains, evidence, actions, primary


def build_recommendation(r, local):
    domains, evidence, actions, primary = operational_reason(r)
    score = num(r, 'priority_score')
    band = str(r['priority_band'])
    m = num(r, 'model_high_risk_score')
    d = num(r, 'delivery_risk_score')
    q = num(r, 'quality_risk_score')
    det = num(r, 'deterioration_score')
    s = num(r, 'strategic_exposure_score')
    p = num(r, 'probability_high')

    if band.lower() == 'critical' or score >= 60 or (m >= 70 and (d >= 60 or q >= 60)):
        level, owner, cadence = 'Immediate intervention', 'Procurement / Supply Chain Leadership', 'Weekly'
    elif band.lower() == 'high' or score >= 40 or det >= 80 or m >= 50 or d >= 60 or q >= 60:
        level, owner, cadence = 'Corrective action', 'Supplier Quality / Procurement', 'Weekly'
    elif band.lower() == 'moderate' or score >= 20 or len(actions) >= 2 or m >= 30 or det >= 70 or d >= 40 or q >= 30:
        level, owner, cadence = 'Enhanced monitoring', 'Supplier Management / Procurement', 'Bi-weekly'
    else:
        level, owner, cadence = 'Routine monitoring', 'Supplier Management', 'Monthly'

    row = {
        'supplier_id': r['supplier_id'],
        'period': r['period'],
        'predicted_risk_N1': r['predicted_risk_N1'],
        'probability_high': p,
        'priority_score': score,
        'priority_band': band,
        'delivery_risk_score': d,
        'quality_risk_score': q,
        'strategic_exposure_score': s,
        'deterioration_score': det,
        'model_high_risk_score': m,
        'intervention_level': level,
        'primary_risk_domain': primary,
        'risk_domains': ' | '.join(dict.fromkeys(domains)),
        'why_risky': '',
        'risk_evidence': ' | '.join(evidence),
        'recommended_intervention': actions[0],
        'additional_interventions': ' | '.join(actions[1:]),
        'action_owner': owner,
        'review_cadence': cadence,
        'requires_corrective_action': level in ['Corrective action', 'Immediate intervention'],
        'requires_continuity_review': 'Strategic exposure' in domains,
        'local_explanation_available': bool(local),
        'top_local_drivers': local.get('top_local_drivers', ''),
        'positive_local_drivers': local.get('positive_local_drivers', ''),
        'opposing_local_drivers': local.get('opposing_local_drivers', ''),
        'source': 'Phase 4 operational score + Phase 5 local explainability when available'
    }

    score_reason = f"Priority score {score:.2f}/100 ({band} band); predicted N+1 risk {r['predicted_risk_N1']}; High-risk probability {p:.2%}."
    operational_reason_text = f"Primary operational concern: {primary}."
    model_reason = ''
    if local:
        model_reason = f" Local model drivers by absolute contribution: {local.get('top_local_drivers','not available')}."
    row['why_risky'] = score_reason + ' ' + operational_reason_text + model_reason
    return row


def main():
    print('=' * 78)
    print('PHASE 6.1: SUPPLIER RISK REASON + EVIDENCE ENHANCEMENT')
    print('=' * 78)

    if not PRIORITY_FILE.exists():
        raise FileNotFoundError(f'Required Phase 4 input not found: {PRIORITY_FILE}')

    df = pd.read_csv(PRIORITY_FILE)
    print(f'Input file: {PRIORITY_FILE}')
    print(f'Input shape: {df.shape}')

    print('\nINPUT VALIDATION')
    missing = [c for c in REQ if c not in df.columns]
    if missing:
        raise ValueError('Missing required Phase 4 columns: ' + ', '.join(missing))
    if df['supplier_id'].duplicated().any():
        raise ValueError('Phase 4 input must contain exactly one latest row per supplier.')
    if set(FUTURE).intersection(df.columns):
        # Presence is allowed in source, but they must not enter operational outputs.
        print('Future/outcome columns detected in source and will be excluded from output evidence.')
    print('Required columns: PASSED')
    print(f'Suppliers: {df.supplier_id.nunique()}')
    print('One row per supplier: PASSED')

    for c in ['priority_score', 'delivery_risk_score', 'quality_risk_score', 'strategic_exposure_score', 'deterioration_score', 'model_high_risk_score']:
        v = pd.to_numeric(df[c], errors='coerce')
        if v.isna().any() or (v < 0).any() or (v > 100).any():
            raise ValueError(f'Invalid score range in {c}')
    p = pd.to_numeric(df['probability_high'], errors='coerce')
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probability_high range')
    print('Score/probability validation: PASSED')

    local_df, expl_status = load_explainability()
    print('\nPHASE 5 EXPLAINABILITY INPUT')
    print(expl_status)
    if not local_df.empty:
        print(f'Usable supplier-level local explanations: {len(local_df)}')
    else:
        print('Usable supplier-level local explanations: 0')

    lookup = {}
    if not local_df.empty:
        for _, x in local_df.iterrows():
            lookup[x['supplier_id']] = x.to_dict()

    out_rows = []
    for _, r in df.iterrows():
        out_rows.append(build_recommendation(r, lookup.get(r['supplier_id'], {})))
    out = pd.DataFrame(out_rows)

    # Hard governance check: no future/outcome columns may appear in operational output.
    forbidden_output = sorted(FUTURE.intersection(out.columns))
    if forbidden_output:
        raise AssertionError('Future/outcome columns leaked into output: ' + ', '.join(forbidden_output))

    if out['supplier_id'].duplicated().any():
        raise AssertionError('Duplicate supplier in Phase 6.1 output')
    if out['why_risky'].isna().any() or out['why_risky'].eq('').any():
        raise AssertionError('Every supplier must have a non-empty why_risky explanation')
    if out['risk_evidence'].isna().any() or out['risk_evidence'].eq('').any():
        raise AssertionError('Every supplier must have evidence')

    order = {'Immediate intervention': 0, 'Corrective action': 1, 'Enhanced monitoring': 2, 'Routine monitoring': 3}
    out['_order'] = out['intervention_level'].map(order)
    out = out.sort_values(['_order', 'priority_score'], ascending=[True, False]).drop(columns='_order').reset_index(drop=True)

    print('\nPHASE 6.1 VALIDATION')
    print('✓ One row per supplier')
    print('✓ Operational score ranges validated')
    print('✓ Future/outcome variables excluded')
    print('✓ Every supplier has an explicit why_risky explanation')
    print('✓ Evidence attached to every supplier')
    print('✓ Phase 5 local drivers attached where schema-compatible data exists')
    print('✓ Deterministic rule-based intervention levels')

    out_path = OUTDIR / 'supplier_intervention_recommendations_explained.csv'
    high_path = OUTDIR / 'high_priority_interventions_explained.csv'
    reason_path = OUTDIR / 'supplier_risk_reasons.csv'
    summary_path = OUTDIR / 'action_summary_explained.csv'
    validation_path = REPORTDIR / 'phase6_1_validation.txt'
    report_path = REPORTDIR / 'phase6_1_risk_reason_summary.json'

    out.to_csv(out_path, index=False)
    out[out['intervention_level'].isin(['Immediate intervention', 'Corrective action'])].to_csv(high_path, index=False)
    out[['supplier_id', 'period', 'predicted_risk_N1', 'probability_high', 'priority_score', 'priority_band',
         'primary_risk_domain', 'risk_domains', 'why_risky', 'risk_evidence', 'top_local_drivers',
         'positive_local_drivers', 'opposing_local_drivers', 'recommended_intervention']].to_csv(reason_path, index=False)
    out.groupby(['intervention_level', 'primary_risk_domain']).size().reset_index(name='supplier_count').to_csv(summary_path, index=False)

    validation_lines = [
        'PHASE 6.1 VALIDATION',
        '====================',
        f'Suppliers: {len(out)}',
        'One row per supplier: PASSED',
        'Operational score ranges: PASSED',
        'Future/outcome exclusion: PASSED',
        'why_risky coverage: PASSED',
        'risk_evidence coverage: PASSED',
        f'Local explainability attached: {int(out.local_explanation_available.sum())}/{len(out)}',
        f'Immediate intervention: {(out.intervention_level == "Immediate intervention").sum()}',
        f'Corrective action: {(out.intervention_level == "Corrective action").sum()}',
        f'Enhanced monitoring: {(out.intervention_level == "Enhanced monitoring").sum()}',
        f'Routine monitoring: {(out.intervention_level == "Routine monitoring").sum()}',
    ]
    validation_path.write_text('\n'.join(validation_lines), encoding='utf-8')

    report = {
        'phase': 'Phase 6.1 - Supplier Risk Reason + Evidence Enhancement',
        'supplier_count': int(len(out)),
        'local_explanation_suppliers': int(out.local_explanation_available.sum()),
        'local_explanation_status': expl_status,
        'governance': {
            'future_outcome_columns_excluded': True,
            'deterministic_rules': True,
            'every_supplier_has_reason': True,
            'every_supplier_has_evidence': True
        },
        'intervention_counts': {k: int(v) for k, v in out['intervention_level'].value_counts().items()},
        'primary_risk_domain_counts': {k: int(v) for k, v in out['primary_risk_domain'].value_counts().items()},
        'outputs': {
            'recommendations_explained': str(out_path),
            'high_priority_explained': str(high_path),
            'supplier_risk_reasons': str(reason_path),
            'action_summary': str(summary_path),
            'validation': str(validation_path)
        }
    }
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('\nTOP SUPPLIERS WITH REASONS')
    cols = ['supplier_id', 'period', 'predicted_risk_N1', 'probability_high', 'priority_score',
            'priority_band', 'primary_risk_domain', 'why_risky', 'recommended_intervention']
    print(out[cols].head(20).to_string(index=False))

    print('\nSAVED:')
    for pth in [out_path, high_path, reason_path, summary_path, validation_path, report_path]:
        print(pth)
    print('\nPHASE 6.1 COMPLETE')
    print('NEXT IMPLEMENTATION STAGE: PHASE 7 - DECISION-SUPPORT DASHBOARD / REPORTING')


if __name__ == '__main__':
    main()
