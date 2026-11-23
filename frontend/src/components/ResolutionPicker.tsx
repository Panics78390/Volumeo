import { formatInt, type Preset, type PresetKey, type Resolution } from '../api'

interface Props {
  presets: Preset[]
  textureSizes: number[]
  facesRange: [number, number]
  value: Resolution
  onChange: (r: Resolution) => void
}

// Curseur logarithmique : 500 → 500 000 triangles reste maniable.
const toSlider = (faces: number, [min, max]: [number, number]) =>
  Math.round((Math.log(faces / min) / Math.log(max / min)) * 1000)
const fromSlider = (v: number, [min, max]: [number, number]) => {
  const faces = min * Math.pow(max / min, v / 1000)
  const step = faces < 10_000 ? 100 : faces < 100_000 ? 1_000 : 5_000
  return Math.min(max, Math.max(min, Math.round(faces / step) * step))
}

/** Choix de la résolution du maillage : préréglages ou réglage personnalisé. */
export default function ResolutionPicker({ presets, textureSizes, facesRange, value, onChange }: Props) {
  const select = (key: PresetKey) => {
    const p = presets.find((x) => x.key === key)
    onChange(p ? { ...value, preset: key, target_faces: p.target_faces, texture_size: p.texture_size } : { ...value, preset: 'custom' })
  }

  return (
    <div>
      <div className="preset-grid">
        {presets.map((p) => (
          <button key={p.key} className={`preset ${value.preset === p.key ? 'active' : ''}`} onClick={() => select(p.key)}>
            <strong>{p.label}</strong>
            <span>{p.description}</span>
          </button>
        ))}
        <button className={`preset ${value.preset === 'custom' ? 'active' : ''}`} onClick={() => select('custom')}>
          <strong>Personnalisée</strong>
          <span>Triangles et texture au choix</span>
        </button>
      </div>

      {value.preset === 'custom' && (
        <div className="field">
          <label htmlFor="faces">
            Nombre de triangles <output>{formatInt(value.target_faces)}</output>
          </label>
          <input
            id="faces"
            type="range"
            min={0}
            max={1000}
            value={toSlider(value.target_faces, facesRange)}
            onChange={(e) => onChange({ ...value, target_faces: fromSlider(Number(e.target.value), facesRange) })}
          />
        </div>
      )}

      <div className="field">
        <label>Taille de la texture</label>
        <div className="segmented">
          {textureSizes.map((s) => (
            <button key={s} className={value.texture_size === s ? 'active' : ''} onClick={() => onChange({ ...value, texture_size: s })}>
              {s} px
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <label htmlFor="smooth">
          Lissage <output>{value.smoothing === 0 ? 'aucun' : `${value.smoothing} passes`}</output>
        </label>
        <input
          id="smooth"
          type="range"
          min={0}
          max={20}
          value={value.smoothing}
          onChange={(e) => onChange({ ...value, smoothing: Number(e.target.value) })}
        />
      </div>

      <label className="check" title="Triangles de taille régulière (PyMeshLab). Plus lent.">
        <input type="checkbox" checked={value.isotropic} onChange={(e) => onChange({ ...value, isotropic: e.target.checked })} />
        Remaillage isotrope (triangles réguliers)
      </label>
    </div>
  )
}
