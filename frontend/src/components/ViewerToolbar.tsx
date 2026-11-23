import type { Backdrop, DisplayMode, ViewSettings } from './Viewer'

interface Props {
  value: ViewSettings
  onChange: (v: ViewSettings) => void
  onResetView: () => void
}

const MODES: { key: DisplayMode; label: string }[] = [
  { key: 'textured', label: 'Texturé' },
  { key: 'clay', label: 'Uni' },
  { key: 'wireframe', label: 'Fil de fer' },
]

const BACKDROPS: { key: Backdrop; label: string }[] = [
  { key: 'dark', label: 'Sombre' },
  { key: 'gray', label: 'Gris' },
  { key: 'light', label: 'Clair' },
]

/** Réglages de la visionneuse : affichage, arêtes, éclairage, fond. */
export default function ViewerToolbar({ value, onChange, onResetView }: Props) {
  const set = <K extends keyof ViewSettings>(k: K, v: ViewSettings[K]) => onChange({ ...value, [k]: v })
  return (
    <div className="toolbar">
      <div className="segmented" role="group" aria-label="Mode d’affichage">
        {MODES.map((m) => (
          <button key={m.key} className={value.mode === m.key ? 'active' : ''} onClick={() => set('mode', m.key)}>
            {m.label}
          </button>
        ))}
      </div>
      <label className="check compact">
        <input type="checkbox" checked={value.edges} disabled={value.mode === 'wireframe'} onChange={(e) => set('edges', e.target.checked)} />
        Arêtes
      </label>
      <label className="check compact">
        <input type="checkbox" checked={value.autoRotate} onChange={(e) => set('autoRotate', e.target.checked)} />
        Rotation auto
      </label>
      <label className="check compact">
        <input type="checkbox" checked={value.grid} onChange={(e) => set('grid', e.target.checked)} />
        Grille
      </label>
      <label className="slider">
        Lumière
        <input type="range" min={0} max={2} step={0.05} value={value.light} onChange={(e) => set('light', Number(e.target.value))} />
      </label>
      <label className="slider">
        Direction
        <input type="range" min={0} max={360} step={5} value={value.lightAngle} onChange={(e) => set('lightAngle', Number(e.target.value))} />
      </label>
      <div className="segmented" role="group" aria-label="Fond">
        {BACKDROPS.map((b) => (
          <button key={b.key} className={value.backdrop === b.key ? 'active' : ''} onClick={() => set('backdrop', b.key)}>
            {b.label}
          </button>
        ))}
      </div>
      <button className="ghost" onClick={onResetView}>
        Recadrer
      </button>
    </div>
  )
}
