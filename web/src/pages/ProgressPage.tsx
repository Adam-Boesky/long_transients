/** Who has labelled what, how the buckets are filling up, and where people
 *  disagree.
 *
 * Laid out like `app_pages/progress.py`, top to bottom:
 *
 *   header         Progress
 *   Labels         grouped horizontal bars, label x member
 *   By member      sources · labels used · share of sample · comments
 *   Disagreements  the sources two people filed differently
 *
 * "Bucket" there is `label` here: the API has no bucket tier, so the
 * vocabulary in `ui.tsx` stands in for the append-only bucket registry.
 *
 * Everything comes from one `/api/progress` call, which cross-tabs the
 * annotations server-side. The Streamlit page did the same work in pandas
 * over a CSV it had already loaded; here the alternative would be shipping
 * every annotation row to the browser to count it.
 */
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import type { Data, Layout } from 'plotly.js'
import { isAbort } from '../api/client'
import { getProgress } from '../api/client'
import type { Disagreement, Progress } from '../api/types'
import { PlotlyChart } from '../charts/PlotlyChart'
import { categorical } from '../charts/palette'
import { PageHeader } from '../components/chrome'
import { Badge, Card } from '../components/ui'

const CHART_CONFIG = { displaylogo: false, responsive: true }

export function ProgressPage() {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    const ac = new AbortController()
    getProgress(ac.signal).then(setProgress)
      .catch((e) => { if (!isAbort(e)) setErr(String(e)) })
    return () => ac.abort()
  }, [])

  // Labels ordered by total use, biggest first. Plotly lays a categorical
  // axis out bottom-up, so the array is reversed to put the busiest bucket
  // at the top of a horizontal chart.
  const labels = useMemo(() => {
    const entries = Object.entries(progress?.by_label ?? {})
    const total = (counts: Record<string, number>) =>
      Object.values(counts).reduce((a, b) => a + b, 0)
    return entries
      .sort((a, b) => total(a[1]) - total(b[1]) || b[0].localeCompare(a[0]))
      .map(([label]) => label)
  }, [progress])

  const data = useMemo<Data[]>(() => {
    if (!progress) return []
    return progress.authors.map((author, i) => ({
      type: 'bar' as const,
      orientation: 'h' as const,
      name: author,
      y: labels,
      x: labels.map((l) => progress.by_label[l]?.[author] ?? 0),
      marker: { color: categorical(i) },
      hovertemplate: `%{y} · ${author}: %{x}<extra></extra>`,
    }))
  }, [progress, labels])

  // Height scales with the number of *bars*, not buckets: the Streamlit
  // version hit the same thing, where sizing by bucket alone made Vega drop
  // y-axis labels to fit.
  const nBars = labels.length * Math.max(1, progress?.authors.length ?? 1)
  const chartHeight = Math.max(220, 30 * nBars + 70)

  const layout = useMemo<Partial<Layout>>(() => ({
    barmode: 'group',
    bargap: 0.3,
    xaxis: { title: { text: 'sources' }, showgrid: true, zeroline: false },
    yaxis: { title: { text: '' }, automargin: true, showgrid: false },
  }), [])

  if (err) return <pre style={{ color: 'crimson' }}>{err}</pre>
  if (!progress) return <div style={{ opacity: 0.6 }}>Loading…</div>

  if (!labels.length) {
    return (
      <>
        <PageHeader title="Progress" large />
        <Card style={{ background: 'var(--bg-soft)', opacity: 0.85 }}>
          No labels assigned yet.
        </Card>
      </>
    )
  }

  return (
    <div style={{ maxWidth: 1080 }}>
      <PageHeader title="Progress" large />

      <Section title="Buckets">
        <Card>
          <div style={{ height: chartHeight }}>
            <PlotlyChart data={data} layout={layout} config={CHART_CONFIG} />
          </div>
        </Card>
      </Section>

      <Section title="By member">
        <MemberTable progress={progress} />
      </Section>

      <Section title="Disagreements">
        {progress.n_disagreements === 0 ? (
          <div style={{ color: 'var(--text-dim)' }}>
            No source has been put in two different labels.
          </div>
        ) : (
          <>
            <div style={{ color: 'var(--text-dim)', marginBottom: 8 }}>
              {progress.n_disagreements.toLocaleString()} source
              {progress.n_disagreements === 1 ? '' : 's'} where members chose
              different labels. The labels column shows the split and the
              comments the reasoning.
              {progress.disagreements.length < progress.n_disagreements
                && ` Showing the ${progress.disagreements.length} most contested.`}
            </div>
            <DisagreementTable rows={progress.disagreements} />
          </>
        )}
      </Section>
    </div>
  )
}

function Section({ title, children }:
                 { title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginBottom: 22 }}>
      <h3 style={{ fontSize: 15, fontWeight: 600, margin: '0 0 8px' }}>
        {title}
      </h3>
      {children}
    </section>
  )
}

const th: React.CSSProperties = {
  textAlign: 'left', padding: '6px 10px', fontWeight: 600,
  color: 'var(--text-dim)', borderBottom: '1px solid var(--border)',
  whiteSpace: 'nowrap',
}

const td: React.CSSProperties = {
  padding: '6px 10px', borderBottom: '1px solid var(--border)',
  verticalAlign: 'top',
}

const num: React.CSSProperties = { ...td, textAlign: 'right' }

function MemberTable({ progress }: { progress: Progress }) {
  return (
    <Card pad={0} style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={th}>Member</th>
            <th style={{ ...th, textAlign: 'right' }}>Sources</th>
            <th style={{ ...th, textAlign: 'right' }}>Buckets used</th>
            <th style={{ ...th, textAlign: 'right' }}>Share of sample</th>
            <th style={{ ...th, textAlign: 'right' }}>Comments</th>
          </tr>
        </thead>
        <tbody>
          {progress.by_member.map((m) => (
            <tr key={m.author}>
              <td style={td}>
                {/* Keyed off the same author ordering the traces use, so the
                    swatch matches the member's bars. */}
                <span style={{ display: 'inline-block', width: 9, height: 9,
                               borderRadius: 2, marginRight: 7,
                               background: categorical(
                                 progress.authors.indexOf(m.author)) }} />
                {m.author}
              </td>
              <td style={num}>{m.n_sources.toLocaleString()}</td>
              <td style={num}>{m.n_labels}</td>
              <td style={num}>
                {progress.n_sources
                  ? `${(100 * m.n_sources / progress.n_sources).toFixed(1)}%`
                  : '—'}
              </td>
              <td style={num}>{m.n_comments.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}

const TONES = ['blue', 'green', 'violet', 'orange', 'red', 'yellow']

function DisagreementTable({ rows }: { rows: Disagreement[] }) {
  // Tone by label, assigned once across the whole table: a label that means
  // the same thing in two rows has to look the same in both.
  const tone = useMemo(() => {
    const seen = [...new Set(rows.flatMap((r) => r.labels))].sort()
    return (label: string) => TONES[seen.indexOf(label) % TONES.length]
  }, [rows])

  return (
    <Card pad={0} style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            <th style={th} />
            <th style={th}>Source</th>
            <th style={{ ...th, textAlign: 'right' }}>RA</th>
            <th style={{ ...th, textAlign: 'right' }}>Dec</th>
            <th style={th}>Buckets</th>
            <th style={{ ...th, width: '45%' }}>Comments</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.source_uuid}>
              <td style={td}>
                {/* Same destination as the Streamlit table's "open" link:
                    the Classify view, showing this source. Routed rather
                    than a plain href so the app is not reloaded. */}
                <Link to={`/classify?page=${encodeURIComponent(r.source_uuid)}`}>
                  open
                </Link>
              </td>
              <td style={{ ...td, fontFamily: 'var(--mono)' }}>
                {r.coord_slug ?? r.source_uuid.slice(0, 8)}
              </td>
              <td style={num}>{r.ra?.toFixed(5) ?? ''}</td>
              <td style={num}>{r.dec?.toFixed(5) ?? ''}</td>
              <td style={td}>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {r.verdicts.filter((v) => v.label).map((v) => (
                    <Badge key={v.author} tone={tone(v.label as string)}>
                      {v.author}: {v.label}
                    </Badge>
                  ))}
                </div>
              </td>
              <td style={td}>
                {r.verdicts.filter((v) => (v.note ?? '').trim())
                  .map((v) => (
                    <div key={v.author} style={{ marginBottom: 2 }}>
                      <b>{v.author}:</b> {v.note}
                    </div>
                  ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  )
}
