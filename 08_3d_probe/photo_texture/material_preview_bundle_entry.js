// Bundle separately from the v5 viewer. Requires three@0.160.1 and esbuild.
// Example (run from a directory with those dependencies installed):
// esbuild material_preview_bundle_entry.js --bundle --minify --format=iife --outfile=material-preview-three-bundle.js
import {
  Scene, PerspectiveCamera, WebGLRenderer, BufferGeometry, Float32BufferAttribute,
  Vector2, Vector3, Group, MeshBasicMaterial, MeshPhysicalMaterial, Mesh,
  CircleGeometry, BoxGeometry, Color, Texture, PMREMGenerator,
  SRGBColorSpace, NoColorSpace, DoubleSide, BackSide, ACESFilmicToneMapping,
  LinearFilter, LinearMipmapLinearFilter
} from 'three';

window.MATERIAL_THREE = {
  Scene, PerspectiveCamera, WebGLRenderer, BufferGeometry, Float32BufferAttribute,
  Vector2, Vector3, Group, MeshBasicMaterial, MeshPhysicalMaterial, Mesh,
  CircleGeometry, BoxGeometry, Color, Texture, PMREMGenerator,
  SRGBColorSpace, NoColorSpace, DoubleSide, BackSide, ACESFilmicToneMapping,
  LinearFilter, LinearMipmapLinearFilter
};
