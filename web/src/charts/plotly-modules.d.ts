/** plotly.js 4.x ships types for its main entry only.
 *
 * `lib/core` and the per-trace modules have no declarations, so the partial
 * bundle needs them declared here. `lib/core` is the same API surface as the
 * main entry minus the trace registry, so it borrows those types wholesale.
 */
declare module 'plotly.js/lib/core' {
  import type * as Plotly from 'plotly.js'
  const core: typeof Plotly & {
    register(modules: unknown | unknown[]): void
  }
  export default core
}

declare module 'plotly.js/lib/scatter' {
  const trace: unknown
  export default trace
}

declare module 'plotly.js/lib/scattergl' {
  const trace: unknown
  export default trace
}
