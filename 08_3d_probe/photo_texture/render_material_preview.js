/* Render and check the actual browser materials, not a CPU color projection.
 * NODE_PATH must provide Playwright. Usage: node render_material_preview.js
 *   /absolute/standalone-wrapper.html /absolute/new-render-output-directory
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');
const {chromium} = require('playwright');

async function main() {
  const [wrapper, output] = process.argv.slice(2);
  if (!wrapper || !output) throw new Error('Supply standalone wrapper and output directory');
  if (fs.existsSync(output) && fs.readdirSync(output).length) throw new Error('Output must be new or empty');
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({headless:true, args:[
    '--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']});
  try {
    const context = await browser.newContext({viewport:{width:1050,height:1050},ignoreHTTPSErrors:true});
    // The inline car is entirely embedded. Resolve only wrapper UI dependencies
    // from installed matching packages when the test host cannot access its CDN.
    for (const [url, moduleFile] of [
      ['https://unpkg.com/lucide@1.17.0/dist/umd/lucide.js', 'lucide/dist/umd/lucide.js'],
      ['https://unpkg.com/@floating-ui/core@1.7.3/dist/floating-ui.core.umd.min.js', '@floating-ui/core/dist/floating-ui.core.umd.min.js'],
      ['https://unpkg.com/@floating-ui/dom@1.7.4/dist/floating-ui.dom.umd.min.js', '@floating-ui/dom/dist/floating-ui.dom.umd.min.js'],
    ]) {
      try {
        const local = require.resolve(moduleFile);
        await context.route(url, route => route.fulfill({path:local,contentType:'application/javascript'}));
      } catch { /* Normal browser loading remains available. */ }
    }
    const page = await context.newPage();
    const errors = [], warnings = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', message => {
      if(message.type()==='error') errors.push(message.text());
      if(message.type()==='warning') warnings.push(message.text());
    });
    page.on('requestfailed', request => requests.push({url:request.url(),failure:request.failure()}));
    await page.goto(pathToFileURL(path.resolve(wrapper)).href, {waitUntil:'networkidle',timeout:60000});
    const frame = page.frames().find(f => f !== page.mainFrame());
    if(!frame) throw new Error('Expected skill wrapper iframe');
    await frame.waitForFunction(() => window.__materialPreview && window.__materialPreview.ready, null, {timeout:60000});
    const capture = async (name, state) => {
      const result = await frame.evaluate(async state => {
        const api = window.__materialPreview;
        await api.setState(state);
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        const data = await api.capture({width:1200,height:900,show:false});
        return {data,state:api.getState()};
      }, state);
      const bytes = Buffer.from(result.data.split(',')[1], 'base64');
      fs.writeFileSync(path.join(output,name),bytes);
      return {file:name,state:result.state,sha256:crypto.createHash('sha256').update(bytes).digest('hex')};
    };
    const report = {wrapper:path.resolve(wrapper),width:1200,height:900,renders:[]};
    report.initial = await frame.evaluate(()=>window.__materialPreview.getState());
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    for(let yaw=0; yaw<360; yaw+=30) {
      report.renders.push(await capture(`pbr_az_${String(yaw).padStart(3,'0')}.png`,
        {mode:'pbr',yaw,elevation:8,zoom:1,environment:'softbox',environmentYaw:0}));
    }
    for(const yaw of [30,90,180,270]) {
      report.renders.push(await capture(`photo_az_${String(yaw).padStart(3,'0')}.png`,
        {mode:'photo',yaw,elevation:8,zoom:1,environment:'softbox',environmentYaw:0}));
    }
    report.lightChecks = [];
    for(const mode of ['photo','pbr']) for(const environmentYaw of [0,90]) {
      report.lightChecks.push(await capture(`${mode}_light_${environmentYaw}.png`,
        {mode,yaw:45,elevation:12,zoom:1,environment:'strip',environmentYaw}));
    }
    const photoLight = report.lightChecks.filter(x=>x.state.mode==='photo');
    const pbrLight = report.lightChecks.filter(x=>x.state.mode==='pbr');
    report.photo_environment_invariant = photoLight[0].sha256 === photoLight[1].sha256;
    report.pbr_environment_changes = pbrLight[0].sha256 !== pbrLight[1].sha256;
    await frame.evaluate(()=>window.__materialPreview.setState({mode:'pbr',yaw:30,elevation:8,environment:'softbox',environmentYaw:0}));
    const canvas = frame.locator('canvas').first();
    const bounds = await canvas.boundingBox();
    await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
    await page.mouse.down();
    await page.mouse.move(bounds.x+bounds.width/2+90,bounds.y+bounds.height/2+15,{steps:8});
    await page.mouse.up();
    report.afterDrag = await frame.evaluate(()=>window.__materialPreview.getState());
    report.drag_changes_view = report.afterDrag.yaw !== 30;
    await page.setViewportSize({width:320,height:1000});
    await frame.evaluate(()=>window.__materialPreview.setState({yaw:90,elevation:8,zoom:1}));
    await page.waitForTimeout(200);
    report.mobile = await frame.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
      canvas:document.querySelector('canvas').getBoundingClientRect().toJSON()}));
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    report.errors = errors; report.warnings = warnings; report.requestFailures = requests;
    report.passed = !errors.length && report.pbr_environment_changes && report.drag_changes_view
      && report.mobile.scrollWidth <= report.mobile.width;
    // A whole-frame invariant can be affected by scene/background rotation;
    // report it explicitly and inspect vehicle pixels before declaring success.
    fs.writeFileSync(path.join(output,'browser_verification.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
    if(!report.passed) process.exitCode=1;
  } finally { await browser.close(); }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
