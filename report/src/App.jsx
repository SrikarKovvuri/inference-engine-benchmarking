import './App.css'
import Hideable from './Hideable.jsx'
import DistributionBoxplot from './DistributionBoxplot.jsx'
import MetricsPlot from './MetricsPlot.jsx'
import DATA from './results.js'

function systemsById(systems) {
  return Object.fromEntries(systems.map((s) => [s.id, s]))
}

function App() {
  const meta = DATA.meta
  const systems = DATA.systems ?? []
  const byId = systemsById(systems)

  return (
    <div className="App">
      <header className="report-header">
        <p className="eyebrow">N4AI · Midterm report</p>
        <h1>{meta.title}</h1>
        <p className="meta-line">
          {meta.uni} · {meta.name} · @{meta.github} · {meta.date}
        </p>
        <p className="meta-line subtle">
          Model: {meta.model} · {meta.device}
        </p>
      </header>

      <main className="report-main">
        <section className="callout takeaways" id="takeaways">
          <h2>Takeaways</h2>
          <ol>
            {(DATA.takeaways ?? []).map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ol>
        </section>

        <section className="callout summary" id="summary">
          <h2>Summary</h2>
          <p>{DATA.summary}</p>
        </section>

        <Hideable title="Results" sectionId="results" defaultOpen>
          <p className="section-lead">
            <strong>Line chart</strong> (required) — JSON <code>plot</code>; metric /
            aggregate / Y-axis pills, legend toggles, hover and click-to-pin.{' '}
            <strong>Boxplot</strong> (required) — JSON <code>boxplot</code>; per-client
            TTFT, TBT, or KV bytes samples; use Distribution / Concurrency pills on that figure.
          </p>
          <h3 className="results-subhead">1. Line chart (required)</h3>
          <MetricsPlot data={DATA} systemsById={byId} />
          <h3 className="results-subhead">2. Boxplot (required)</h3>
          {DATA.boxplot ? (
            <DistributionBoxplot data={DATA} systemsById={byId} />
          ) : (
            <p className="plot-error">
              Missing top-level <code>boxplot</code> in your results JSON (required).
              Add <code>ttft_ms_samples</code> / <code>tbt_ms_samples</code> on concurrent
              runs — see README.
            </p>
          )}
        </Hideable>

        <Hideable title="Systems compared" sectionId="systems" defaultOpen={false}>
          <ul className="systems-list">
            {systems.map((s) => (
              <li key={s.id}>
                <strong>{s.label}</strong>{' '}
                <span className="pill">{s.family}</span>
                <p>{s.description}</p>
                {s.repo ? (
                  <p className="system-repo">
                    <a href={s.repo} target="_blank" rel="noreferrer">
                      {s.repo.replace(/^https:\/\//, '')}
                    </a>
                  </p>
                ) : null}
                {s.novelty ? (
                  <p className="system-novelty">
                    <strong>Key novelty:</strong> {s.novelty}
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        </Hideable>

        <Hideable title="Measurement setup" sectionId="setup" defaultOpen={false}>
          <dl className="setup-dl">
            <dt>Harness</dt>
            <dd>
              <code>{DATA.source.harness}</code>
            </dd>
            <dt>Results JSON</dt>
            <dd>
              <code>{DATA.source.json_file}</code>
            </dd>
            <dt>Wall clock</dt>
            <dd>
              {DATA.source.started} → {DATA.source.finished}
            </dd>
            <dt>Workload</dt>
            <dd>
              Prompt targets{' '}
              {DATA.source.workload.prompt_token_targets.join(', ')} tokens ·{' '}
              {DATA.source.workload.max_new_tokens} new tokens · concurrency{' '}
              {DATA.source.workload.concurrency_grid.join(', ')} ·{' '}
              {DATA.source.workload.reps_per_cell} reps/cell
            </dd>
          </dl>
          <p className="section-lead">{DATA.source.workload.notes}</p>
        </Hideable>

        <Hideable title="Discussion" sectionId="discussion" defaultOpen={false}>
          <p className="placeholder">
            Replace this paragraph in <code>src/App.jsx</code> (or add a{' '}
            <code>discussion</code> field to your JSON and map it here). Tie
            each system’s <code>novelty</code> field to your plots (Orca
            batching tails, vLLM paging, llama.cpp runtime, prefix reuse, PD if
            you ran it). Explain when TTFT vs TBT vs KV bytes tell different
            stories.
          </p>
        </Hideable>

        <Hideable title="Raw run count" sectionId="raw" defaultOpen={false}>
          <p>
            {(DATA.runs ?? []).length} rows in <code>runs[]</code> (commit the
            full JSON; the page aggregates medians for the plot).
          </p>
        </Hideable>
      </main>

      <footer className="report-footer">
        <p>
          Template: edit <code>report/src/results.js</code>, your{' '}
          <code>&lt;UNI&gt;-midterm-results.json</code>, and sections in{' '}
          <code>report/src/App.jsx</code>. Push to <code>main</code> for GitHub
          Pages.
        </p>
      </footer>
    </div>
  )
}

export default App
