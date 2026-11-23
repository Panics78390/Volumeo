import { useRef, useState, type DragEvent } from 'react'
import type { Upload } from '../api'

const ACCEPT = ['image/jpeg', 'image/png', 'image/webp']

const METHOD_LABEL: Record<Upload['background_method'], string> = {
  rembg: 'Fond supprimé par IA (rembg)',
  alpha: 'Transparence d’origine conservée',
  fallback: 'Fond supprimé (détourage simple par couleur)',
  none: 'Arrière-plan conservé',
}

interface Props {
  upload: Upload | null
  busy: boolean
  removeBg: boolean
  maxMb: number
  onRemoveBgChange: (v: boolean) => void
  onFile: (file: File) => void
  onError: (message: string) => void
  onClear: () => void
}

/** Zone de glisser-déposer + aperçu de l'image avant/après détourage. */
export default function Dropzone({ upload, busy, removeBg, maxMb, onRemoveBgChange, onFile, onError, onClear }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [showOriginal, setShowOriginal] = useState(false)

  const pick = (file: File | undefined) => {
    if (!file) return
    if (!ACCEPT.includes(file.type)) return onError('Format non supporté. Utilisez une image JPG, PNG ou WebP.')
    if (file.size > maxMb * 1024 * 1024) return onError(`Image trop lourde (maximum ${maxMb} Mo).`)
    setShowOriginal(false)
    onFile(file)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    pick(e.dataTransfer.files[0])
  }

  return (
    <div>
      <div
        className={`dropzone ${over ? 'over' : ''} ${upload ? 'filled' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="Importer une image"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        <input
          ref={input}
          type="file"
          accept={ACCEPT.join(',')}
          hidden
          onChange={(e) => {
            pick(e.target.files?.[0])
            e.target.value = ''
          }}
        />
        {busy ? (
          <div className="dropzone-hint">
            <div className="spinner" />
            <p>Analyse de l’image et détourage…</p>
          </div>
        ) : upload ? (
          <img
            className="dropzone-preview checker"
            src={showOriginal ? upload.urls.original : upload.urls.cutout}
            alt={showOriginal ? 'Image d’origine' : 'Image détourée'}
          />
        ) : (
          <div className="dropzone-hint">
            <svg viewBox="0 0 24 24" width="34" height="34" aria-hidden="true">
              <path d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v3a1 1 0 001 1h14a1 1 0 001-1v-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <p>
              <strong>Glissez une image ici</strong>
              <br />
              ou cliquez pour parcourir
            </p>
            <small>JPG, PNG ou WebP · {maxMb} Mo max</small>
          </div>
        )}
      </div>

      {upload && !busy && (
        <div className="upload-meta">
          <span title={upload.filename}>
            {upload.width} × {upload.height} px · {METHOD_LABEL[upload.background_method]}
          </span>
          <div className="row">
            <button className="link" onClick={() => setShowOriginal((v) => !v)}>
              {showOriginal ? 'Voir le détourage' : 'Voir l’original'}
            </button>
            <button className="link" onClick={onClear}>
              Retirer
            </button>
          </div>
        </div>
      )}

      <label className="check">
        <input type="checkbox" checked={removeBg} onChange={(e) => onRemoveBgChange(e.target.checked)} />
        Supprimer l’arrière-plan automatiquement
      </label>
    </div>
  )
}
