import { useRef } from 'react'

/** The same value back, with an identity that only changes when its *contents*
 *  do.
 *
 *  Query objects here are rebuilt by the render that sends them, so an effect
 *  depending on one refetches whenever anything else on the page re-renders.
 *  Three places worked around that by serialising the object into the
 *  dependency list and parsing it back inside the effect, which compares
 *  correctly but leaves the effect holding a string where it wants the object.
 *
 *  Comparing by JSON rather than shallowly because these carry arrays
 *  (`bucket_in`, `status_out`) that are rebuilt alongside the object; a
 *  shallow compare would see two equal lists as different and change nothing.
 *  The inputs are small flat query objects, so the cost is a rounding error
 *  against the request it prevents.
 *
 *  Key order matters, as it does for any JSON compare: these are built by one
 *  literal each, so the order is fixed by the source.
 */
export function useStableValue<T>(value: T): T {
  const key = JSON.stringify(value)
  // Written during render, which is safe because it is idempotent: the same
  // key always yields the same stored value, so a re-run cannot disagree.
  const ref = useRef({ key, value })
  if (ref.current.key !== key) ref.current = { key, value }
  return ref.current.value
}
