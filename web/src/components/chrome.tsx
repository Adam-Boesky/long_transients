/** Page chrome: the nav rail and the header row.
 *
 * Mirrors the Streamlit app. `streamlit_app.py` renders navigation and then
 * lets each page emit its own header, so the title and the account control
 * share one centred row -- `chrome.page_header` exists precisely because
 * putting the account menu in the entry point left it on a different line.
 * Same split here: `AppShell` owns the rail, each page renders `PageHeader`.
 */
import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { getAuthors, isAbort } from '../api/client'
import { FilterProvider } from '../lib/filterState'

const NAV = [
  { to: '/classify', label: 'Classify', icon: '🔍' },
  { to: '/table', label: 'Table', icon: '▤' },
  { to: '/progress', label: 'Progress', icon: '📈' },
]

export function AppShell({ member, onSignOut }:
                         { member: string; onSignOut: () => void }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr',
                  minHeight: '100vh', font: '13px system-ui' }}>
      {/* Pinned to the viewport and exactly one screen tall, so the account
          block can sit at the bottom of the screen however far the page
          scrolls. */}
      <nav style={{ borderRight: '1px solid var(--border)',
                    padding: '14px 10px', minWidth: 132,
                    position: 'sticky', top: 0, height: '100vh',
                    alignSelf: 'start', boxSizing: 'border-box',
                    display: 'flex', flexDirection: 'column' }}>
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} style={({ isActive }) => ({
            display: 'block', padding: '7px 10px', marginBottom: 3,
            borderRadius: 6, textDecoration: 'none', color: 'inherit',
            background: isActive ? 'var(--bg-soft)' : 'transparent',
            fontWeight: isActive ? 600 : 400,
          })}>
            <span style={{ marginRight: 7, opacity: 0.8 }}>{n.icon}</span>
            {n.label}
          </NavLink>
        ))}
        <div style={{ marginTop: 'auto', padding: '10px 8px 0',
                      opacity: 0.6, fontSize: 12 }}>
          <div>{member}</div>
          <button onClick={onSignOut}
                  style={{ marginTop: 4, font: 'inherit', cursor: 'pointer',
                           background: 'none', border: 'none', padding: 0,
                           color: 'inherit', textDecoration: 'underline' }}>
            switch user
          </button>
        </div>
      </nav>
      {/* The sidebar's filters live above the routes, so switching between
          Classify and Table keeps the sample you narrowed to rather than
          quietly handing you a different one. */}
      <main style={{ padding: '14px 18px', minWidth: 0 }}>
        <FilterProvider member={member}><Outlet /></FilterProvider>
      </main>
    </div>
  )
}

/** Title left, account right, on one centred row. `beforeTitle` and
 *  `afterTitle` sit immediately either side of the title text (e.g. a tag,
 *  a copy-link icon). */
export function PageHeader({ title, large, right, beforeTitle, afterTitle }:
                           { title: string; large?: boolean;
                             right?: React.ReactNode
                             beforeTitle?: React.ReactNode
                             afterTitle?: React.ReactNode }) {
  const Tag = large ? 'h1' : 'h2'
  // The title group never shrinks below its content (a coord slug can't
  // break), so when the row runs out of room `right` wraps onto its own line
  // instead of being painted over.
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12,
                  flexWrap: 'wrap', marginBottom: 10 }}>
      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6 }}>
        {beforeTitle}
        <Tag style={{ margin: 0, fontSize: large ? 24 : 17, fontWeight: 600 }}>
          {title}
        </Tag>
        {afterTitle}
      </div>
      {right}
    </div>
  )
}

/** The identity gate. `require_member` in the Streamlit app runs before
 *  navigation so the picker is the whole screen, not a page.
 *
 *  Existing members come first, most active at the top: picking yourself
 *  from a list is how you avoid filing half your work as "adam" and half as
 *  "Adam". The free-text field stays for someone new. */
export function MemberGate({ onPick }: { onPick: (name: string) => void }) {
  const [authors, setAuthors] = useState<[string, number][] | null>(null)
  useEffect(() => {
    const ac = new AbortController()
    getAuthors(ac.signal)
      .then((a) => setAuthors(Object.entries(a)
        .sort(([x, m], [y, n]) => n - m || x.localeCompare(y))))
      .catch((e) => { if (!isAbort(e)) { console.error(e); setAuthors([]) } })
    return () => ac.abort()
  }, [])

  const box = { padding: '7px 9px', font: 'inherit', borderRadius: 5,
                border: '1px solid var(--border-strong)' }
  return (
    <div style={{ display: 'grid', placeItems: 'center', minHeight: '100vh',
                  font: '13px system-ui' }}>
      <div style={{ display: 'grid', gap: 10, minWidth: 260 }}>
        <div style={{ font: '600 16px system-ui' }}>Who are you?</div>
        <div style={{ opacity: 0.65 }}>
          Your name is stored with every classification.
        </div>
        {authors === null && <div style={{ opacity: 0.4 }}>·</div>}
        {authors && authors.length > 0 && (
          <div style={{ display: 'grid', gap: 5 }}>
            {authors.map(([name]) => (
              <button key={name} onClick={() => onPick(name)}
                      style={{ ...box, cursor: 'pointer', textAlign: 'left',
                               background: 'transparent', color: 'inherit' }}>
                {name}
              </button>
            ))}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault()
            const v = new FormData(e.currentTarget).get('member')
            if (typeof v !== 'string' || !v.trim()) return
            // "adam" when "Adam" exists means Adam, not a second member.
            const typed = v.trim()
            const known = authors?.find(
              ([name]) => name.toLowerCase() === typed.toLowerCase())
            onPick(known ? known[0] : typed)
          }}
          style={{ display: 'grid', gap: 10,
                   marginTop: authors?.length ? 6 : 0 }}
        >
          {authors && authors.length > 0 && (
            <div style={{ opacity: 0.65 }}>or someone new</div>
          )}
          <input name="member" autoFocus placeholder="name"
                 style={{ ...box, background: 'var(--bg)', color: 'inherit' }} />
          <button type="submit" style={{ ...box, cursor: 'pointer' }}>
            Continue
          </button>
        </form>
      </div>
    </div>
  )
}
