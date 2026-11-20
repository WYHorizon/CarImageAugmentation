/* Render the published GLB with the same daylight as the repair viewer. */
const fs = require('node:fs/promises');
const path = require('node:path');
const {chromium} = require('playwright');
const esbuild = require('esbuild');

async function main() {
  const [out, option, value] = process.argv.slice(2);
  if (!out || (option && !['--street', '--angle'].includes(option)) ||
      (option === '--angle' && (!value || !Number.isFinite(Number(value))))) {
    throw new Error('Usage: node render.cjs OUTPUT [--street | --angle DEGREES]');
  }
  let views;
  if (option === '--street') {
    const cameraPath = path.join(__dirname, '../../07_background_replacement/bmw_street/camera.json');
    const camera = JSON.parse(await fs.readFile(cameraPath, 'utf8'));
    views = [{...camera, name:'shopping-2x', width:2160, height:1622, supersampling:2}];
  } else {
    const angles = option === '--angle' ? [Number(value)] : [32, 90, 148];
    views = angles.map(azimuth => ({azimuth, elevation:8, width:1024, height:660,
                                     name:'view-' + azimuth}));
  }
  const glb = await fs.readFile(path.join(__dirname, 'bmw5.glb'));
  const script = await esbuild.build({
    entryPoints:[path.join(__dirname, 'scene.js')], bundle:true, write:false,
    nodePaths:(process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean)
  });
  const browser = await chromium.launch({args:['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const page = await browser.newPage();
    await page.addScriptTag({content:script.outputFiles[0].text});
    const frames = await page.evaluate(([bytes, cameras]) => window.renderModel(bytes, cameras),
                                       [glb.toString('base64'), views]);
    await fs.mkdir(out, {recursive:true});
    for (const [i, frame] of frames.entries()) {
      await fs.writeFile(path.join(out, views[i].name + '.png'), Buffer.from(frame, 'base64'));
    }
  } finally {
    await browser.close();
  }
}
main().catch(error => {console.error(error); process.exitCode = 1;});
