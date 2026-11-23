import { Component, Suspense, useEffect, useMemo, type ReactNode } from 'react'
import { Canvas } from '@react-three/fiber'
import { Bounds, Grid, OrbitControls, useGLTF } from '@react-three/drei'
import * as THREE from 'three'

export type DisplayMode = 'textured' | 'clay' | 'wireframe'
export type Backdrop = 'dark' | 'gray' | 'light'

export interface ViewSettings {
  mode: DisplayMode
  edges: boolean // superpose les arêtes (fil de fer) sur la surface
  autoRotate: boolean
  grid: boolean
  light: number // intensité globale de l'éclairage (0 – 2)
  lightAngle: number // azimut de la lumière principale, en degrés
  backdrop: Backdrop
}

export const defaultViewSettings: ViewSettings = {
  mode: 'textured',
  edges: false,
  autoRotate: false,
  grid: true,
  light: 1,
  lightAngle: 40,
  backdrop: 'dark',
}

export interface ModelStats {
  triangles: number
  vertices: number
  textureSize: number | null
}

const clayMaterial = new THREE.MeshStandardMaterial({ color: '#b9bfcc', roughness: 0.75, metalness: 0.05 })
const wireMaterial = new THREE.MeshBasicMaterial({ color: '#4da3ff', wireframe: true })
const edgeMaterial = new THREE.MeshBasicMaterial({
  color: '#0b0d14',
  wireframe: true,
  transparent: true,
  opacity: 0.35,
  depthWrite: false,
})

interface Part {
  geometry: THREE.BufferGeometry
  material: THREE.Material
  matrix: THREE.Matrix4
}

function Model({ url, settings, onStats }: { url: string; settings: ViewSettings; onStats: (s: ModelStats) => void }) {
  const gltf = useGLTF(url)

  // On extrait les maillages du GLB pour contrôler nous-mêmes leurs matériaux.
  const parts = useMemo<Part[]>(() => {
    const out: Part[] = []
    gltf.scene.updateMatrixWorld(true)
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh
      if (m.isMesh) out.push({ geometry: m.geometry, material: m.material as THREE.Material, matrix: m.matrixWorld.clone() })
    })
    return out
  }, [gltf])

  useEffect(() => {
    let triangles = 0
    let vertices = 0
    let textureSize: number | null = null
    for (const p of parts) {
      const pos = p.geometry.getAttribute('position')
      vertices += pos.count
      triangles += (p.geometry.index ? p.geometry.index.count : pos.count) / 3
      const map = (p.material as THREE.MeshStandardMaterial).map
      const img = map?.image as { width?: number } | undefined
      if (img?.width) textureSize = img.width
    }
    onStats({ triangles, vertices, textureSize })
  }, [parts, onStats])

  const material = (p: Part) =>
    settings.mode === 'textured' ? p.material : settings.mode === 'clay' ? clayMaterial : wireMaterial

  return (
    <group>
      {parts.map((p, i) => (
        <group key={i} matrix={p.matrix} matrixAutoUpdate={false}>
          <mesh geometry={p.geometry} material={material(p)} />
          {settings.edges && settings.mode !== 'wireframe' && (
            <mesh geometry={p.geometry} material={edgeMaterial} renderOrder={1} />
          )}
        </group>
      ))}
    </group>
  )
}

function Lights({ intensity, angle }: { intensity: number; angle: number }) {
  const a = (angle * Math.PI) / 180
  const key: [number, number, number] = [Math.sin(a) * 3, 2.5, Math.cos(a) * 3]
  const fill: [number, number, number] = [-key[0], 1, -key[2]]
  return (
    <>
      <ambientLight intensity={0.55 * intensity} />
      <hemisphereLight args={['#ffffff', '#30364a', 0.6 * intensity]} />
      <directionalLight position={key} intensity={2.2 * intensity} />
      <directionalLight position={fill} intensity={0.5 * intensity} />
    </>
  )
}

class ViewerErrorBoundary extends Component<{ children: ReactNode; resetKey: string }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(e: Error) {
    return { error: e.message }
  }
  componentDidUpdate(prev: { resetKey: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null })
  }
  render() {
    if (this.state.error)
      return (
        <div className="viewer-empty">
          <p>Impossible d’afficher le modèle.</p>
          <small>{this.state.error}</small>
        </div>
      )
    return this.props.children
  }
}

interface Props {
  url: string | null
  settings: ViewSettings
  viewKey: number // changer la valeur recadre la caméra
  onStats: (s: ModelStats) => void
  placeholder?: ReactNode
}

export default function Viewer({ url, settings, viewKey, onStats, placeholder }: Props) {
  if (!url) return <div className="viewer-empty">{placeholder}</div>
  return (
    <ViewerErrorBoundary resetKey={url}>
      <Canvas
        className={`viewer-canvas backdrop-${settings.backdrop}`}
        camera={{ position: [1.4, 1.1, 2.2], fov: 40, near: 0.01, far: 100 }}
        dpr={[1, 2]}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
      >
        <Lights intensity={settings.light} angle={settings.lightAngle} />
        {/* L'indicateur de chargement est affiché par le parent (en HTML, hors du canvas). */}
        <Suspense fallback={null}>
          <Bounds key={`${url}-${viewKey}`} fit clip observe margin={1.25}>
            <Model url={url} settings={settings} onStats={onStats} />
          </Bounds>
        </Suspense>
        {settings.grid && (
          <Grid
            position={[0, -0.001, 0]}
            infiniteGrid
            cellSize={0.1}
            sectionSize={0.5}
            cellColor={settings.backdrop === 'light' ? '#b6bccb' : '#2a3042'}
            sectionColor={settings.backdrop === 'light' ? '#8b93a8' : '#3d4662'}
            fadeDistance={12}
            fadeStrength={2}
          />
        )}
        <OrbitControls
          makeDefault
          enableDamping
          autoRotate={settings.autoRotate}
          autoRotateSpeed={1.5}
          minDistance={0.2}
          maxDistance={20}
        />
      </Canvas>
    </ViewerErrorBoundary>
  )
}
