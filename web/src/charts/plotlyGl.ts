/** The WebGL trace, registered on the shared Plotly instance.
 *
 * Imported only by `SpectrumChart`, which the classify page loads on demand,
 * so the WebGL stack (regl and friends) is a separate chunk that the Table
 * and Progress pages never download. Registration is global: once this has
 * run, `scattergl` works from the same `Plotly` every other chart uses.
 *
 * Kept off the light curves on purpose: a browser allows only a handful of
 * WebGL contexts (Plotly hits trouble past ~8 charts on a page), and the
 * spectrum is the only trace big enough to need it.
 */
import Plotly from './plotly'
import scattergl from 'plotly.js/lib/scattergl'

Plotly.register([scattergl])
