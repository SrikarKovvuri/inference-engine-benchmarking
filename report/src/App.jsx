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
          <div className="discussion">
            <p>
              The line chart and boxplot measure the same TinyLlama 1.1B math
              under three different schedulers. TTFT is where those designs show
              up. TBT and KV bytes tell a quieter story: decode stays roughly
              stable per engine, and the dense KV footprint grows linearly with
              sequence length on all three systems.
            </p>

            <h3>Baseline: one worker, a FIFO queue, no batching</h3>
            <p>
              <code>minimal_full_server</code> is still the Week 2 five-step
              loop. Concurrent clients are just threads that arrive together and
              sit in a queue. The worker finishes an entire request — prefill
              plus all 64 decode steps — before it dequeues the next one. There
              is no iteration-level batching and the KV cache is a contiguous
              Hugging Face <code>DynamicCache</code>.
            </p>
            <p>
              That novelty is a staircase, not a faster GPU kernel. At
              concurrency 1 and 512 tokens, median TTFT is 449 ms and TBT p50 is
              about 51 ms. At concurrency 4 the same TBT stays near 50 ms, but
              median TTFT jumps to 6280 ms and the slowest client in the boxplot
              waits 12044 ms. The first client looks like the N=1 run; each later
              client pays for every earlier request&apos;s full prefill and
              decode. The boxplot whiskers are the point: the tail is the design.
            </p>

            <h3>vLLM: PagedAttention plus continuous batching</h3>
            <p>
              vLLM keeps the same FP16 TinyLlama weights but changes memory
              layout and admission. PagedAttention stores KV in blocks and maps
              them with a block table, so a request does not need one growing
              contiguous tensor. That reduces fragmentation and wasted reserved
              space, which is what lets the engine keep more live sequences on
              an 8 GB GPU. It does not shrink the information in a full cache:
              our KV-bytes series uses the same dense GQA formula for every
              engine, and baseline and vLLM match at 4.3 / 13.0 / 44.7 MB for
              192 / 576 / 1984 tokens. Paging changes allocation, not the
              logical footprint at the end of the run.
            </p>
            <p>
              Continuous batching is the latency result. A new request does not
              wait for an in-flight decode to finish; it can join the next
              iteration after its own prefill. At N=4 and 512 tokens, vLLM&apos;s
              median TTFT is 1562 ms and the worst client sample is 1638 ms —
              about 4× the N=1 median of 412 ms, not the baseline&apos;s 14×
              jump to 12 s. TBT stays near the single-user line (~33–36 ms)
              because decode steps are shared. The boxplot is tight compared
              with the baseline whiskers: the remaining TTFT spread is shared
              prefill and scheduling, not a serialized queue.
            </p>

            <h3>llama.cpp: C++/GGUF runtime, slots, and a Q8 confounder</h3>
            <p>
              llama.cpp is a different runtime, not just a different scheduler.
              It serves a Q8_0 GGUF through a C++ CUDA <code>llama-server</code>
              with four parallel slots and an 8192-token shared KV pool (so each
              of four clients can hold a 2048-token context). Slot-based
              continuous batching is why N=4 does not collapse: 512-token median
              TTFT is 72 ms and the worst client sample is 79 ms.
            </p>
            <p>
              The line chart looks almost flat because those numbers are not an
              FP16 apples-to-apples comparison. Q8 cuts weight bandwidth versus
              the Hugging Face FP16 models used by baseline and vLLM. After a
              same-prompt warmup, long-context N=1 TTFT stays 34–60 ms versus
              1627 / 1393 ms on the FP16 engines, which is consistent with
              prefix reuse in addition to the faster runtime. The right reading
              is: the C++ server plus slot batching keeps concurrency cheap, and
              quantization / cache reuse explain most of the absolute gap. TBT
              still rises from ~22 ms at N=1 to ~33 ms at N=4, so the four slots
              do share the GPU — they just do not queue whole requests the way
              the baseline does.
            </p>

            <h3>What the two figures disagree on</h3>
            <p>
              The N=1 line chart says all three engines get slower as prompts
              get longer, which is prefill. The N=4 boxplot says the baseline is
              a different system: its tail is queue delay, vLLM&apos;s tail is
              shared-batch delay, and llama.cpp&apos;s box barely moves. KV
              bytes do not separate the engines because we reported the dense
              formula, not vLLM&apos;s paged allocator. The scheduling novelty
              lives in TTFT tails, not in tok/s or in the KV line.
            </p>
          </div>
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
