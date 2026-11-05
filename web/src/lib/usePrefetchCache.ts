/** Warms the browser's HTTP cache with the *next* source's cutouts.
 *
 * This is what makes paging feel instant: by the time you press Next the
 * images are already cached and the `<img>` resolves from memory.
 *
 * Lives here rather than in `CutoutGrid` because nothing in that component
 * uses it -- the grid draws the current source, while the page decides which
 * source to warm next, and only the page ever called this.
 */
import { useCallback, useRef } from 'react'
import type { CutoutImages } from '../components/CutoutGrid'

/** Keeps the most recent prefetches alive.
 *
 * A browser may evict an image that has no live reference before it is used.
 * Holding the last few `Image` objects prevents that, bounded so a long
 * session cannot accumulate them.
 */
export function usePrefetchCache(limit = 24) {
  const held = useRef<HTMLImageElement[]>([])
  // `useCallback` for the same reason the window object is memoised: callers
  // depend on this in effects.
  return useCallback((images: CutoutImages | null | undefined) => {
    if (!images) return
    for (const bands of Object.values(images)) {
      for (const url of Object.values(bands)) {
        const img = new Image()
        img.decoding = 'async'
        img.src = url
        held.current.push(img)
      }
    }
    if (held.current.length > limit) {
      held.current = held.current.slice(-limit)
    }
  }, [limit])
}
