import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
const {DataTexture, EquirectangularReflectionMapping, FloatType, LinearFilter,
       LinearSRGBColorSpace, PMREMGenerator, RGBAFormat} = THREE;

function daylight(renderer){
  const width=256, height=128, pixels=new Float32Array(width*height*4);
  for(let y=0; y<height; y++){
    const latitude=Math.PI*((y+.5)/height-.5), up=Math.sin(latitude), radius=Math.cos(latitude);
    for(let x=0; x<width; x++){
      const longitude=2*Math.PI*((x+.5)/width-.5);
      const east=radius*Math.cos(longitude), north=radius*Math.sin(longitude);
      const sky=.5+.5*Math.tanh((up+.045)/.11);
      const ambient=.16+sky*(.77+.54*Math.max(0,up));
      // Continuous overcast sky with a broad patch of brighter daylight.
      // No panorama or photo of real objects is projected onto the glass.
      const broad=2.8*Math.exp(7.2*(-.78*east+.47*up+.40*north-1));
      const fill=.6*Math.exp(4*(.65*east+.70*up-.30*north-1));
      const i=(y*width+x)*4;
      pixels[i]=ambient*(1.02-.07*sky)+broad+fill;
      pixels[i+1]=ambient+broad*.99+fill;
      pixels[i+2]=ambient*(.96+.07*sky)+broad*.97+fill*1.03;
      pixels[i+3]=1;
    }
  }
  const texture=new DataTexture(pixels,width,height,RGBAFormat,FloatType);
  texture.colorSpace=LinearSRGBColorSpace;
  texture.mapping=EquirectangularReflectionMapping;
  texture.magFilter=texture.minFilter=LinearFilter;
  texture.needsUpdate=true;
  const pmrem=new PMREMGenerator(renderer);
  const environment=pmrem.fromEquirectangular(texture);
  pmrem.dispose();texture.dispose();
  return environment;
}


window.renderModel = async function (encoded, views) {
  const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true, preserveDrawingBuffer:true});
  renderer.setClearColor(0, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  const scene = new THREE.Scene();
  const sky = daylight(renderer);
  scene.environment = sky.texture;
  const bytes = Uint8Array.from(atob(encoded), ch => ch.charCodeAt(0));
  const model = (await new GLTFLoader().parseAsync(bytes.buffer, '')).scene;
  scene.add(model);
  const bounds = new THREE.Box3().setFromObject(model);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const camera = new THREE.PerspectiveCamera(27, 1, .01, 100);
  const frames = [];
  for (const view of views) {
    const {width, height} = view;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (view.vanishing_point) {
      model.rotation.y = THREE.MathUtils.degToRad(view.yaw_deg);
      model.position.set(view.car_origin_x, 0, -view.camera_distance);
      camera.position.set(0, view.camera_y, 0);
      camera.rotation.set(0, 0, 0);
      const [cx, cy] = view.vanishing_point;
      const n = camera.near, f = view.lens, s = view.supersampling;
      camera.projectionMatrix.makePerspective(-n*cx/f, n*(width/s-cx)/f, n*cy/f, -n*(height/s-cy)/f, n, camera.far);
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    } else {
      model.rotation.set(0,0,0); model.position.set(0,0,0);
      const az = THREE.MathUtils.degToRad(view.azimuth);
      const el = THREE.MathUtils.degToRad(view.elevation);
      const direction = new THREE.Vector3(Math.sin(az)*Math.cos(el), Math.sin(el), Math.cos(az)*Math.cos(el));
      camera.position.copy(center).addScaledVector(direction, size.z*1.7);
      camera.lookAt(center);
    }
    renderer.render(scene, camera);
    frames.push(renderer.domElement.toDataURL('image/png').split(',')[1]);
  }
  sky.dispose();
  renderer.dispose();
  return frames;
};
