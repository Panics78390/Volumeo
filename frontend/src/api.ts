// Client HTTP typé de l'API FastAPI (toutes les routes sont sous /api).

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
export type PresetKey = 'draft' | 'standard' | 'high' | 'custom'
export type ExportFormat = 'glb' | 'obj' | 'stl' | 'fbx'

export interface Preset {
  key: Exclude<PresetKey, 'custom'>
  label: string
  description: string
  target_faces: number
  texture_size: number
  mc_resolution: number
}

export interface GeneratorInfo {
  name: string
  label: string
  description: string
  requires_gpu: boolean
  available: boolean
  reason: string
}

export interface SystemInfo {
  device: {
    torch: boolean
    cuda: boolean
    device: string
    gpu_name: string | null
    gpu_memory_gb: number | null
    warning: string | null
  }
  generators: GeneratorInfo[]
  default_generator: string
  presets: Preset[]
  texture_sizes: number[]
  faces_range: [number, number]
  formats: Record<ExportFormat, { filename: string; description: string }>
  max_upload_mb: number
  text_to_image: { available: boolean; reason: string; model: string; max_prompt_chars: number }
}

export interface Upload {
  id: string
  filename: string
  width: number
  height: number
  background_method: 'rembg' | 'alpha' | 'fallback' | 'none'
  urls: { original: string; cutout: string }
}

export interface JobParams {
  generator: string
  target_faces: number
  texture_size: number
  mc_resolution: number
  smoothing: number
  isotropic: boolean
  prompt: string | null
  source_upload_id: string | null
}

export interface JobResult {
  faces: number
  vertices: number
  watertight: boolean
  size: [number, number, number]
  raw_faces: number
  timings: Record<string, number>
}

export interface Job {
  id: string
  name: string
  kind: 'generate' | 'reprocess' | 'text'
  status: JobStatus
  progress: number
  message: string
  params: JobParams
  result: JobResult | null
  error: string | null
  upload_id: string | null
  parent_id: string | null
  created_at: number
  started_at: number | null
  finished_at: number | null
  queue_position: number | null
  urls: {
    preview: string
    model: string | null
    texture: string | null
    downloads: Partial<Record<ExportFormat, string>>
  }
}

export interface Resolution {
  preset: PresetKey
  target_faces: number
  texture_size: number
  smoothing: number
  isotropic: boolean
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch {
    throw new Error('Serveur injoignable. Le backend est-il démarré ?')
  }
  if (!res.ok) {
    let detail = `Erreur ${res.status}`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join(' ; ')
    } catch {
      /* réponse non JSON */
    }
    throw new Error(detail)
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T)
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

function resolutionBody(r: Resolution) {
  return {
    preset: r.preset,
    // Les préréglages imposent leur nombre de triangles ; seule la texture reste réglable.
    target_faces: r.preset === 'custom' ? r.target_faces : undefined,
    texture_size: r.texture_size,
    smoothing: r.smoothing,
    isotropic: r.isotropic,
  }
}

export const api = {
  system: () => request<SystemInfo>('/api/system'),

  upload(file: File, removeBg: boolean) {
    const form = new FormData()
    form.append('file', file)
    form.append('remove_bg', String(removeBg))
    return request<Upload>('/api/uploads', { method: 'POST', body: form })
  },

  /** Image seule, description seule, ou les deux (la description transforme alors l'image). */
  createJob: (source: { uploadId?: string; prompt?: string; strength?: number }, generator: string, r: Resolution) =>
    request<Job>(
      '/api/jobs',
      json({
        upload_id: source.uploadId,
        prompt: source.prompt || undefined,
        strength: source.strength,
        generator,
        ...resolutionBody(r),
      }),
    ),

  reprocess: (jobId: string, r: Resolution) => request<Job>(`/api/jobs/${jobId}/reprocess`, json(resolutionBody(r))),

  jobs: () => request<{ total: number; items: Job[] }>('/api/jobs?limit=100'),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/api/jobs/${id}`, { method: 'DELETE' }),
}

export const isActive = (j: Job) => j.status === 'queued' || j.status === 'running'

export const formatInt = (n: number) => n.toLocaleString('fr-FR')
