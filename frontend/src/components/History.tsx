import { formatInt, isActive, type Job } from '../api'

const STATUS: Record<Job['status'], string> = {
  queued: 'En attente',
  running: 'En cours',
  done: 'Terminé',
  failed: 'Échec',
  cancelled: 'Annulé',
}

const timeFmt = new Intl.DateTimeFormat('fr-FR', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' })

interface Props {
  jobs: Job[]
  selectedId: string | null
  onSelect: (job: Job) => void
  onDelete: (job: Job) => void
  onCancel: (job: Job) => void
}

/** Historique des générations (les plus récentes en premier). */
export default function History({ jobs, selectedId, onSelect, onDelete, onCancel }: Props) {
  if (jobs.length === 0) return <p className="muted">Vos générations apparaîtront ici.</p>
  return (
    <ul className="history">
      {jobs.map((job) => (
        <li key={job.id} className={`history-item ${job.id === selectedId ? 'selected' : ''}`}>
          <button className="history-main" onClick={() => onSelect(job)}>
            {/* L'aperçu n'existe qu'une fois la génération démarrée : `key` force un nouvel essai à chaque changement d'état. */}
            {job.status === 'queued' ? (
              <div className="thumb checker" />
            ) : (
              <img key={job.status} className="thumb checker" src={job.urls.preview} alt="" loading="lazy" />
            )}
            <div className="history-text">
              <strong title={job.name}>{job.name}</strong>
              <span>
                {job.result ? `${formatInt(job.result.faces)} triangles` : `~${formatInt(job.params.target_faces)} triangles`} ·{' '}
                {job.params.texture_size} px
              </span>
              <span className="muted">
                {timeFmt.format(new Date(job.created_at * 1000))}
                {job.kind === 'reprocess' && ' · ajustement'}
              </span>
              {isActive(job) ? (
                <div className="mini-progress" aria-label={`${Math.round(job.progress * 100)} %`}>
                  <div style={{ width: `${Math.round(job.progress * 100)}%` }} />
                </div>
              ) : (
                <span className={`badge ${job.status}`}>{STATUS[job.status]}</span>
              )}
            </div>
          </button>
          {isActive(job) ? (
            <button className="icon" title="Annuler" aria-label={`Annuler ${job.name}`} onClick={() => onCancel(job)}>
              ✕
            </button>
          ) : (
            <button className="icon" title="Supprimer" aria-label={`Supprimer ${job.name}`} onClick={() => onDelete(job)}>
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
                <path d="M4 7h16M9 7V4h6v3m-8 0l1 13h8l1-13" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
        </li>
      ))}
    </ul>
  )
}
