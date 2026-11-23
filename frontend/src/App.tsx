import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, formatInt, isActive, type ExportFormat, type Job, type Resolution, type SystemInfo, type Upload } from './api'
import Dropzone from './components/Dropzone'
import History from './components/History'
import ResolutionPicker from './components/ResolutionPicker'
import Viewer, { defaultViewSettings, type ModelStats, type ViewSettings } from './components/Viewer'
import ViewerToolbar from './components/ViewerToolbar'

const FORMAT_LABEL: Record<ExportFormat, string> = { glb: 'GLB', obj: 'OBJ + MTL', stl: 'STL', fbx: 'FBX' }

const DEFAULT_RESOLUTION: Resolution = { preset: 'standard', target_faces: 50_000, texture_size: 2048, smoothing: 0, isotropic: false }

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [file, setFile] = useState<File | null>(null)
  const [upload, setUpload] = useState<Upload | null>(null)
  const [uploading, setUploading] = useState(false)
  const [removeBg, setRemoveBg] = useState(true)

  const [generator, setGenerator] = useState('')
  const [resolution, setResolution] = useState<Resolution>(DEFAULT_RESOLUTION)
  const [submitting, setSubmitting] = useState(false)

  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const [view, setView] = useState<ViewSettings>(defaultViewSettings)
  const [viewKey, setViewKey] = useState(0)
  const [stats, setStats] = useState<ModelStats | null>(null)

  const fail = useCallback((e: unknown) => setError(e instanceof Error ? e.message : String(e)), [])

  // -- chargement initial ---------------------------------------------------
  useEffect(() => {
    api
      .system()
      .then((s) => {
        setSystem(s)
        setGenerator(s.default_generator)
      })
      .catch(fail)
  }, [fail])

  // -- historique + suivi de progression (polling) -------------------------------
  const refresh = useCallback(async () => {
    try {
      setJobs((await api.jobs()).items)
    } catch (e) {
      fail(e)
    }
  }, [fail])

  const hasActive = jobs.some(isActive)
  useEffect(() => {
    refresh()
    // Rafraîchissement rapide tant qu'une génération tourne, lent sinon.
    const id = setInterval(refresh, hasActive ? 700 : 8000)
    return () => clearInterval(id)
  }, [refresh, hasActive])

  // Sélectionne la génération la plus récente au premier chargement.
  const autoSelected = useRef(false)
  useEffect(() => {
    if (!autoSelected.current && jobs.length > 0) {
      autoSelected.current = true
      setSelectedId((cur) => cur ?? jobs[0].id)
    }
  }, [jobs])

  const selected = useMemo(() => jobs.find((j) => j.id === selectedId) ?? null, [jobs, selectedId])
  const modelUrl = selected?.status === 'done' ? selected.urls.model : null
  useEffect(() => {
    if (!modelUrl) setStats(null)
  }, [modelUrl])

  // -- import ---------------------------------------------------------------
  const sendFile = useCallback(
    async (f: File, withRemoveBg: boolean) => {
      setUploading(true)
      setError(null)
      try {
        setUpload(await api.upload(f, withRemoveBg))
      } catch (e) {
        setUpload(null)
        fail(e)
      } finally {
        setUploading(false)
      }
    },
    [fail],
  )

  const onFile = (f: File) => {
    setFile(f)
    sendFile(f, removeBg)
  }

  const onRemoveBgChange = (v: boolean) => {
    setRemoveBg(v)
    if (file) sendFile(file, v) // re-détoure l'image déjà importée
  }

  // -- génération -------------------------------------------------------------
  const generate = async () => {
    if (!upload) return
    setSubmitting(true)
    setError(null)
    try {
      const job = await api.createJob(upload.id, generator, resolution)
      setJobs((prev) => [job, ...prev])
      setSelectedId(job.id)
    } catch (e) {
      fail(e)
    } finally {
      setSubmitting(false)
    }
  }

  const adjust = async () => {
    if (!selected) return
    setSubmitting(true)
    setError(null)
    try {
      const job = await api.reprocess(selected.id, resolution)
      setJobs((prev) => [job, ...prev])
      setSelectedId(job.id)
    } catch (e) {
      fail(e)
    } finally {
      setSubmitting(false)
    }
  }

  const removeJob = async (job: Job) => {
    try {
      await api.remove(job.id)
      setJobs((prev) => prev.filter((j) => j.id !== job.id))
      if (selectedId === job.id) setSelectedId(null)
    } catch (e) {
      fail(e)
    }
  }

  const cancelJob = async (job: Job) => {
    try {
      await api.cancel(job.id)
      refresh()
    } catch (e) {
      fail(e)
    }
  }

  const gen = system?.generators.find((g) => g.name === generator)
  const device = system?.device

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
            <path d="M16 4l10 5.5v13L16 28 6 22.500v-13z M6 9.500l10 5.500 10-5.500 M16 15v13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          </svg>
          <h1>Volumeo</h1>
        </div>
        {device && (
          <span className={`device ${device.cuda ? 'gpu' : 'cpu'}`} title={device.warning ?? undefined}>
            {device.cuda ? `GPU · ${device.gpu_name} (${device.gpu_memory_gb} Go)` : 'CPU uniquement'}
          </span>
        )}
      </header>

      {error && (
        <div className="alert error" role="alert">
          <span>{error}</span>
          <button className="icon" aria-label="Fermer" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}

      <main className="layout">
        {/* ----------------------------------------------------------------- réglages */}
        <aside className="panel controls">
          <section>
            <h2>
              <span className="step">1</span> Image
            </h2>
            <Dropzone
              upload={upload}
              busy={uploading}
              removeBg={removeBg}
              maxMb={system?.max_upload_mb ?? 20}
              onRemoveBgChange={onRemoveBgChange}
              onFile={onFile}
              onError={setError}
              onClear={() => {
                setUpload(null)
                setFile(null)
              }}
            />
          </section>

          <section>
            <h2>
              <span className="step">2</span> Résolution du maillage
            </h2>
            {system && (
              <ResolutionPicker
                presets={system.presets}
                textureSizes={system.texture_sizes}
                facesRange={system.faces_range}
                value={resolution}
                onChange={setResolution}
              />
            )}
          </section>

          <section>
            <h2>
              <span className="step">3</span> Modèle de génération
            </h2>
            <select value={generator} onChange={(e) => setGenerator(e.target.value)} aria-label="Modèle de génération">
              {system?.generators.map((g) => (
                <option key={g.name} value={g.name} disabled={!g.available}>
                  {g.label}
                  {g.available ? '' : ' — non installé'}
                </option>
              ))}
            </select>
            {gen && <p className="muted small">{gen.description}</p>}
            {gen?.requires_gpu && device && !device.cuda && (
              <p className="alert warn small">Aucun GPU CUDA détecté : ce modèle tournera sur CPU, comptez de 30 s à plusieurs minutes.</p>
            )}
            {system?.generators.some((g) => !g.available) && generator === 'procedural' && (
              <p className="muted small">Pour une vraie reconstruction par IA, installez TripoSR (voir le README).</p>
            )}
          </section>

          <div className="actions">
            <button className="primary" disabled={!upload || uploading || submitting || !gen?.available} onClick={generate}>
              Générer le modèle 3D
            </button>
            <button
              className="secondary"
              disabled={selected?.status !== 'done' || submitting}
              onClick={adjust}
              title="Applique la résolution, la texture et le lissage choisis au modèle affiché, sans relancer la génération"
            >
              Ajuster le modèle affiché
            </button>
          </div>
        </aside>

        {/* ----------------------------------------------------------------- visionneuse */}
        <section className="panel stage">
          <div className="viewer">
            <Viewer
              url={modelUrl}
              settings={view}
              viewKey={viewKey}
              onStats={setStats}
              placeholder={
                selected && isActive(selected) ? (
                  <div className="progress-card">
                    <strong>{selected.status === 'queued' ? 'En file d’attente' : selected.message}</strong>
                    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(selected.progress * 100)}>
                      <div style={{ width: `${Math.round(selected.progress * 100)}%` }} />
                    </div>
                    <span className="muted">
                      {selected.status === 'queued'
                        ? selected.queue_position
                          ? `${selected.queue_position} génération(s) avant celle-ci`
                          : 'Démarrage imminent'
                        : `${Math.round(selected.progress * 100)} %`}
                    </span>
                    <button className="ghost" onClick={() => cancelJob(selected)}>
                      Annuler
                    </button>
                  </div>
                ) : selected?.status === 'failed' ? (
                  <div className="progress-card">
                    <strong>La génération a échoué</strong>
                    <span className="muted">{selected.error}</span>
                  </div>
                ) : selected?.status === 'cancelled' ? (
                  <div className="progress-card">
                    <strong>Génération annulée</strong>
                  </div>
                ) : (
                  <div className="progress-card plain">
                    <strong>Aucun modèle à afficher</strong>
                    <span className="muted">Importez une image, choisissez une résolution puis lancez la génération.</span>
                  </div>
                )
              }
            />
            {modelUrl && !stats && <div className="viewer-loading">Chargement du modèle…</div>}
            {modelUrl && stats && (
              <div className="stats" aria-label="Statistiques du modèle">
                <span>
                  <b>{formatInt(stats.triangles)}</b> triangles
                </span>
                <span>
                  <b>{formatInt(stats.vertices)}</b> sommets
                </span>
                {stats.textureSize && (
                  <span>
                    texture <b>{stats.textureSize} px</b>
                  </span>
                )}
              </div>
            )}
          </div>

          <ViewerToolbar value={view} onChange={setView} onResetView={() => setViewKey((k) => k + 1)} />

          {selected?.status === 'done' && selected.result && (
            <div className="result">
              <div className="result-info">
                <strong>{selected.name}</strong>
                <span className="muted">
                  {system?.generators.find((g) => g.name === selected.params.generator)?.label ?? selected.params.generator} · généré en{' '}
                  {selected.result.timings.total.toLocaleString('fr-FR')} s ·{' '}
                  {selected.result.watertight ? 'maillage étanche (imprimable)' : 'maillage non étanche (à réparer avant impression)'}
                </span>
              </div>
              <div className="downloads">
                {(Object.keys(FORMAT_LABEL) as ExportFormat[]).map((fmt) => (
                  <a key={fmt} className="download" href={selected.urls.downloads[fmt]} download title={system?.formats[fmt]?.description}>
                    <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                      <path d="M12 4v12m0 0l-4-4m4 4l4-4M5 20h14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {FORMAT_LABEL[fmt]}
                  </a>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* ----------------------------------------------------------------- historique */}
        <aside className="panel history-panel">
          <h2>Historique</h2>
          <History jobs={jobs} selectedId={selectedId} onSelect={(j) => setSelectedId(j.id)} onDelete={removeJob} onCancel={cancelJob} />
        </aside>
      </main>
    </div>
  )
}
