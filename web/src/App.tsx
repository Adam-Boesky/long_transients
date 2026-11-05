/** Routing and the identity gate.
 *
 * The member picker runs *before* navigation, as `streamlit_app.py` does, so
 * it occupies the whole screen rather than appearing as one page among three.
 * The catalog schema is fetched once here: it is stable across rebuilds and
 * every page that shows catalog columns needs it.
 */
import { useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { getCatalogSchema, isAbort } from './api/client'
import type { CatalogSchema } from './api/types'
import { AppShell, MemberGate } from './components/chrome'
import { ClassifyPage } from './pages/ClassifyPage'
import { ProgressPage } from './pages/ProgressPage'
import { TablePage } from './pages/TablePage'

const MEMBER_KEY = 'lt.member'

export default function App() {
  const [member, setMember] = useState<string | null>(
    () => localStorage.getItem(MEMBER_KEY))
  const [schema, setSchema] = useState<CatalogSchema | null>(null)

  useEffect(() => {
    if (!member) return
    const ac = new AbortController()
    getCatalogSchema(ac.signal).then(setSchema)
      .catch((e) => { if (!isAbort(e)) console.error(e) })
    return () => ac.abort()
  }, [member])

  if (!member) {
    return <MemberGate onPick={(name) => {
      localStorage.setItem(MEMBER_KEY, name)
      setMember(name)
    }} />
  }

  const signOut = () => { localStorage.removeItem(MEMBER_KEY); setMember(null) }

  return (
    <Routes>
      <Route element={<AppShell member={member} onSignOut={signOut} />}>
        <Route path="/classify"
               element={<ClassifyPage member={member} schema={schema} />} />
        <Route path="/table"
               element={<TablePage member={member} />} />
        <Route path="/progress"
               element={<ProgressPage />} />
        <Route path="*" element={<Navigate to="/classify" replace />} />
      </Route>
    </Routes>
  )
}
