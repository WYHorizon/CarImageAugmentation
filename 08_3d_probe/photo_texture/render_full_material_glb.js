/* Offline validation of the exported GLB with the standard Three GLTFLoader.
 * Dependencies: three@0.160.1, esbuild, playwright; expose them via NODE_PATH.
 * Example:
 * NODE_PATH=/tmp/car-3d-viewer-check/node_modules \
 * PLAYWRIGHT_BROWSERS_PATH=/tmp/car-3d-viewer-check/browsers \
 * node render_full_material_glb.js /absolute/material-output-directory
 * The input GLB is read-only. The output directory must be new or empty.
 */
const fs = require('fs');
const path = require('path');
const os = require('os');
const http = require('http');
const crypto = require('crypto');
const esbuild = require('esbuild');
const { chromium } = require('playwright');

const hash = data => crypto.createHash('sha256').update(data).digest('hex');

async function main() {
  const [assetDirectory, outputDirectory] = process.argv.slice(2);
  if (!assetDirectory) throw new Error('Supply the asset directory containing material_mesh.glb');
  const asset = path.resolve(assetDirectory);
  const source = path.join(asset, 'material_mesh.glb');
  const output = path.resolve(outputDirectory || path.join(asset, 'full_glb_renders'));
  if (fs.existsSync(output) && fs.readdirSync(output).length) throw new Error('Output must be new or empty');
  const glb = fs.readFileSync(source);
  fs.mkdirSync(output, { recursive: true });
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'car-full-glb-'));
  const bundlePath = path.join(temporary, 'full-glb-three.js');
  const entry = fs.readFileSync(path.join(__dirname, 'material_preview_bundle_entry.js'), 'utf8')
    + '\nimport {GLTFLoader} from "three/examples/jsm/loaders/GLTFLoader.js"; window.MaterialGLTFLoader=GLTFLoader;\n';
  await esbuild.build({
    stdin: { contents: entry, resolveDir: __dirname, sourcefile: 'full-glb-entry.js' },
    nodePaths: (process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean),
    bundle: true, minify: true, format: 'iife', outfile: bundlePath
  });
  const bundle = fs.readFileSync(bundlePath);
  // Reuse the exact procedural environment from the interactive source rather
  // than silently evaluating this GLB under unrelated illumination.
  const template = fs.readFileSync(path.join(__dirname, 'material_preview_template.html'), 'utf8');
  const start = template.indexOf('    function studioEnvironment(kind) {');
  const end = template.indexOf('    function capture(options = {}) {', start);
  if (start < 0 || end < 0) throw new Error('Cannot locate the preview environment helper');
  const environmentFunction = template.slice(start, end);
  const server = http.createServer((request, response) => {
    if (request.url === '/full-glb-three.js') {
      response.writeHead(200, { 'Content-Type': 'application/javascript' }); response.end(bundle);
    } else if (request.url === '/material_mesh.glb') {
      response.writeHead(200, { 'Content-Type': 'model/gltf-binary', 'Content-Length': glb.length }); response.end(glb);
    } else if (request.url === '/favicon.ico') { response.writeHead(204); response.end();
    } else if (request.url === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end('<!doctype html><meta charset="utf-8"><title>Offline GLB validation</title><script src="/full-glb-three.js"></script>');
    } else { response.writeHead(404); response.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  const report = { source, sourceSha256: hash(glb), sourceBytes: glb.length, renders: [] };
  const errors = [], warnings = [], failedRequests = [];
  try {
    browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
    const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); if (message.type() === 'warning') warnings.push(message.text()); });
    page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
    await page.goto('http://127.0.0.1:' + server.address().port + '/', { waitUntil: 'networkidle' });
    const browserScript = `(async () => {
      const T = window.MATERIAL_THREE;
      const scene = new T.Scene();
      const renderer = new T.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
      renderer.setPixelRatio(1); renderer.setSize(1200,900);
      renderer.outputColorSpace=T.SRGBColorSpace; renderer.toneMapping=T.ACESFilmicToneMapping; renderer.toneMappingExposure=1;
      renderer.setClearColor(new T.Color().setRGB(1,1,1,T.SRGBColorSpace),1);
      document.body.style.margin='0'; document.body.appendChild(renderer.domElement);
      const camera = new T.PerspectiveCamera(35,4/3,.01,100);
      const gltf = await new window.MaterialGLTFLoader().loadAsync('/material_mesh.glb');
      const loaded = gltf.scene, meshes = [];
      loaded.traverse(object=>{if(object.isMesh)meshes.push(object)});
      const minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];
      loaded.updateMatrixWorld(true);
      for(const mesh of meshes){
        const position=mesh.geometry.getAttribute('position');
        for(let i=0;i<position.count;i++){
          const point=new T.Vector3().fromBufferAttribute(position,i).applyMatrix4(mesh.matrixWorld);
          for(let j=0;j<3;j++){const value=[point.x,point.y,point.z][j];minimum[j]=Math.min(minimum[j],value);maximum[j]=Math.max(maximum[j],value)}
        }
      }
      const center=minimum.map((v,i)=>(v+maximum[i])/2);
      loaded.position.set(-center[0],-center[1],-center[2]);
      const model=new T.Group(); model.add(loaded);scene.add(model);
      const corners=[];
      for(const x of [minimum[0]-center[0],maximum[0]-center[0]])for(const y of [minimum[1]-center[1],maximum[1]-center[1]])for(const z of [minimum[2]-center[2],maximum[2]-center[2]])corners.push(new T.Vector3(x,y,z));
      const radius=Math.sqrt(maximum.reduce((sum,v,i)=>sum+Math.pow((v-minimum[i])/2,2),0));
      const groundMaterial=new T.MeshBasicMaterial({color:new T.Color().setScalar(.94),side:T.DoubleSide,toneMapped:false});
      const ground=new T.Mesh(new T.CircleGeometry(radius*1000,32),groundMaterial);ground.rotation.x=-Math.PI/2;ground.position.y=minimum[1]-center[1]-radius*.015;scene.add(ground);
      const pmrem=new T.PMREMGenerator(renderer);
      ${environmentFunction}
      const environments={softbox:studioEnvironment('softbox'),strip:studioEnvironment('strip')};
      const textureInfo=texture=>texture?{width:texture.image.width,height:texture.image.height,colorSpace:texture.colorSpace,flipY:texture.flipY,channel:texture.channel}:null;
      const meshInfo=meshes.map(mesh=>{
        const material=mesh.material,normal=mesh.geometry.getAttribute('normal');let minNormal=Infinity,maxNormal=0,nonfinite=0;
        for(let i=0;i<normal.count;i++){const length=Math.hypot(normal.getX(i),normal.getY(i),normal.getZ(i));if(!Number.isFinite(length))nonfinite++;else{minNormal=Math.min(minNormal,length);maxNormal=Math.max(maxNormal,length)}}
        // Match the inline viewer's handling of photo colors; this changes the
        // display tone mapping only, never the GLB or its material extensions.
        if(material.isMeshBasicMaterial){material.toneMapped=false;mesh.renderOrder=1}
        return {name:mesh.name,material:material.name,type:material.type,vertices:mesh.geometry.getAttribute('position').count,indices:mesh.geometry.index.count,
          alphaTest:material.alphaTest,transparent:material.transparent,side:material.side,roughness:material.roughness,metalness:material.metalness,
          clearcoat:material.clearcoat,clearcoatRoughness:material.clearcoatRoughness,normalLength:{minimum:minNormal,maximum:maxNormal,nonfinite},
          map:textureInfo(material.map),roughnessMap:textureInfo(material.roughnessMap),metalnessMap:textureInfo(material.metalnessMap),clearcoatMap:textureInfo(material.clearcoatMap),
          sharedControlTexture:material.isMeshPhysicalMaterial?material.clearcoatMap===material.roughnessMap&&material.roughnessMap===material.metalnessMap:null};
      });
      const sameArray=(a,b)=>a.length===b.length&&a.every((value,i)=>value===b[i]);
      const sameGeometry=meshes.length===2&&['position','normal','uv'].every(name=>sameArray(meshes[0].geometry.getAttribute(name).array,meshes[1].geometry.getAttribute(name).array))&&sameArray(meshes[0].geometry.index.array,meshes[1].geometry.index.array);
      const physical=meshes.find(mesh=>mesh.material.isMeshPhysicalMaterial),detail=meshes.find(mesh=>mesh.material.isMeshBasicMaterial);
      let alphaComplement=null;
      if(physical&&detail){
        const imageA=physical.material.map.image,imageB=detail.material.map.image;
        if(imageA.width===imageB.width&&imageA.height===imageB.height){
          const canvas=document.createElement('canvas');canvas.width=imageA.width;canvas.height=imageA.height;const context=canvas.getContext('2d',{willReadFrequently:true});
          context.drawImage(imageA,0,0);const a=context.getImageData(0,0,canvas.width,canvas.height).data;context.clearRect(0,0,canvas.width,canvas.height);context.drawImage(imageB,0,0);const b=context.getImageData(0,0,canvas.width,canvas.height).data;
          let bad=0,pbr=0,unlit=0;for(let i=3;i<a.length;i+=4){if(a[i]+b[i]!==255)bad++;if(a[i]>=128)pbr++;if(b[i]>=128)unlit++}alphaComplement={badPixels:bad,pbrPixels:pbr,unlitPixels:unlit,totalPixels:a.length/4};
        }
      }
      const degrees=Math.PI/180;
      window.__fullGLB={info:{meshes:meshInfo,sameGeometry,alphaComplement,extensionsUsed:gltf.parser.json.extensionsUsed,viewer:{unlitToneMapping:false,exposure:1,toneMapping:'ACESFilmic',environment:'template procedural studio'}},
        capture(state){
          const yaw=(state.yaw??30)*degrees,elevation=(state.elevation??8)*degrees,environmentAngle=-(state.environmentYaw??0)*degrees;
          model.rotation.y=environmentAngle;scene.environment=environments[state.environment||'softbox'].texture;
          const right=new T.Vector3(Math.cos(yaw),0,-Math.sin(yaw)),up=new T.Vector3(-Math.sin(yaw)*Math.sin(elevation),Math.cos(elevation),-Math.cos(yaw)*Math.sin(elevation)),view=new T.Vector3(Math.sin(yaw)*Math.cos(elevation),Math.sin(elevation),Math.cos(yaw)*Math.cos(elevation));
          const halfVertical=camera.fov*degrees/2,halfHorizontal=Math.atan(Math.tan(halfVertical)*camera.aspect);let fit=0;
          for(const corner of corners)fit=Math.max(fit,corner.dot(view)+Math.max(Math.abs(corner.dot(right))/Math.tan(halfHorizontal),Math.abs(corner.dot(up))/Math.tan(halfVertical)));
          const distance=fit*1.12;camera.position.set(distance*Math.sin(yaw+environmentAngle)*Math.cos(elevation),distance*Math.sin(elevation),distance*Math.cos(yaw+environmentAngle)*Math.cos(elevation));camera.lookAt(0,0,0);renderer.render(scene,camera);return renderer.domElement.toDataURL('image/png');
        }};
      window.__fullGLB.capture({yaw:30});
      return window.__fullGLB.info;
    })()`;
    report.loaded = await page.evaluate(browserScript);
    for (const state of [
      { yaw: 30, elevation: 8, environment: 'softbox', environmentYaw: 0 },
      { yaw: 90, elevation: 8, environment: 'softbox', environmentYaw: 0 },
      { yaw: 45, elevation: 12, environment: 'strip', environmentYaw: 0 },
      { yaw: 45, elevation: 12, environment: 'strip', environmentYaw: 90 }
    ]) {
      const data = await page.evaluate(state => window.__fullGLB.capture(state), state);
      const bytes = Buffer.from(data.split(',')[1], 'base64');
      const file = state.environment === 'softbox' ? `full_pbr_az_${String(state.yaw).padStart(3, '0')}.png` : `full_pbr_light_${state.environmentYaw}.png`;
      fs.writeFileSync(path.join(output, file), bytes);
      report.renders.push({ file, state, width: 1200, height: 900, sha256: hash(bytes) });
    }
    report.pbrEnvironmentChanges = report.renders[2].sha256 !== report.renders[3].sha256;
    const types = report.loaded.meshes.map(mesh => mesh.type).sort();
    report.passed = !errors.length && !failedRequests.length && report.loaded.sameGeometry
      && JSON.stringify(types) === JSON.stringify(['MeshBasicMaterial', 'MeshPhysicalMaterial'])
      && report.loaded.alphaComplement?.badPixels === 0 && report.pbrEnvironmentChanges
      && report.loaded.meshes.every(mesh => mesh.alphaTest === .5 && mesh.map?.width > 0 && !mesh.normalLength.nonfinite
        && mesh.normalLength.minimum > .99 && mesh.normalLength.maximum < 1.01)
      && report.loaded.meshes.find(mesh => mesh.type === 'MeshPhysicalMaterial')?.sharedControlTexture;
  } catch (error) { report.passed = false; errors.push(error.stack || error.message); }
  finally {
    report.errors = errors; report.warnings = warnings; report.failedRequests = failedRequests;
    fs.writeFileSync(path.join(output, 'full_glb_verification.json'), JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
    if (browser) await browser.close(); await new Promise(resolve => server.close(resolve));
    fs.rmSync(temporary, { recursive: true, force: true });
  }
  if (!report.passed) process.exitCode = 1;
}

main().catch(error => { console.error(error); process.exitCode = 1; });
