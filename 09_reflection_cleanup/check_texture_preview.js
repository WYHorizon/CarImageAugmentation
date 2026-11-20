const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {pathToFileURL} = require('url');
const {chromium} = require('playwright');

async function main() {
  const [wrapper,output] = process.argv.slice(2);
  if(!wrapper || !output || (fs.existsSync(output)&&fs.readdirSync(output).length)) throw new Error('Use wrapper and fresh output directory');
  fs.mkdirSync(output,{recursive:true});
  const browser = await chromium.launch({headless:true,args:['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  try {
    const context=await browser.newContext({viewport:{width:800,height:1050},ignoreHTTPSErrors:true,colorScheme:'light'});
    for(const [url,moduleFile] of [
      ['https://unpkg.com/lucide@1.17.0/dist/umd/lucide.js','lucide/dist/umd/lucide.js'],
      ['https://unpkg.com/@floating-ui/core@1.7.3/dist/floating-ui.core.umd.min.js','@floating-ui/core/dist/floating-ui.core.umd.min.js'],
      ['https://unpkg.com/@floating-ui/dom@1.7.4/dist/floating-ui.dom.umd.min.js','@floating-ui/dom/dist/floating-ui.dom.umd.min.js']]) {
      const local=path.join(process.env.NODE_PATH.split(path.delimiter)[0],moduleFile);
      if(!fs.existsSync(local))throw new Error('Missing bundled browser dependency: '+local);
      await context.route(url,route=>route.fulfill({path:local,contentType:'application/javascript'}));
    }
    const page=await context.newPage(),errors=[],failed=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',e=>{if(e.type()==='error')errors.push(e.text());});
    page.on('requestfailed',r=>failed.push({url:r.url(),failure:r.failure()}));
    await page.goto(pathToFileURL(path.resolve(wrapper)).href,{waitUntil:'networkidle',timeout:60000});
    const frame=page.frames().find(f=>f!==page.mainFrame());
    if(!frame)throw new Error('Missing preview iframe');
    await frame.waitForFunction(()=>{const b=document.getElementById('clean-capture');return b&&!b.disabled;},null,{timeout:60000});
    const report={wrapper:path.resolve(wrapper),wrapper_sha256:crypto.createHash('sha256').update(fs.readFileSync(wrapper)).digest('hex')};
    report.initial_mode=await frame.locator('#clean-mode').inputValue();
    report.available_modes=await frame.locator('#clean-mode option').evaluateAll(options=>options.map(x=>x.value));
    report.metrics=await frame.evaluate(()=>window.__texturePreview?.metrics||null);
    await frame.locator('#clean-yaw').fill('90');
    await frame.locator('#clean-yaw').dispatchEvent('input');
    const capture=async (mode)=>{
      await frame.locator('#clean-mode').selectOption(mode);
      await frame.locator('#clean-capture').click();
      const data=await frame.locator('.capture-image').getAttribute('src');
      await frame.locator('.capture-image').evaluate(im=>im.decode());
      const bytes=Buffer.from(data.split(',')[1],'base64');
      fs.writeFileSync(path.join(output,mode+'_az_090.png'),bytes);
      return {sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
        angle:await frame.locator('#clean-yaw-value').textContent(),
        elevation:await frame.locator('#clean-elevation-value').textContent(),
        camera:await frame.evaluate(()=>window.__texturePreview?.getCamera()||null),
        size:await frame.locator('.capture-image').evaluate(im=>[im.naturalWidth,im.naturalHeight])};
    };
    report.captures={};
    for(const mode of report.available_modes)report.captures[mode]=await capture(mode);
    report.original=report.captures.original;report.clean=report.captures.clean;
    const captures=Object.values(report.captures);
    report.texture_switch_changes_pixels=new Set(captures.map(x=>x.sha256)).size===captures.length;
    report.switch_retains_camera=captures.every(x=>x.angle===report.original.angle&&x.elevation===report.original.elevation&&JSON.stringify(x.camera)===JSON.stringify(report.original.camera));
    report.capture_size_correct=captures.every(x=>x.size[0]===1200&&x.size[1]===900);
    const returned=await capture('original');
    report.roundtrip_original_pixels_exact=returned.sha256===report.original.sha256;
    await frame.locator('#clean-mode').selectOption(report.initial_mode);
    await frame.locator('.capture').evaluate(e=>e.hidden=true);
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    const bounds=await frame.locator('.model-stage canvas').boundingBox();
    await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);
    await page.mouse.down();await page.mouse.move(bounds.x+bounds.width/2+80,bounds.y+bounds.height/2+12,{steps:6});await page.mouse.up();
    report.after_drag=await frame.locator('#clean-yaw-value').textContent();
    report.drag_changes_view=report.after_drag!==report.clean.angle;
    await page.setViewportSize({width:320,height:1000});
    await frame.locator('#clean-yaw').fill('90');await frame.locator('#clean-yaw').dispatchEvent('input');
    await frame.locator('#clean-elevation').fill('2');await frame.locator('#clean-elevation').dispatchEvent('input');
    await page.waitForTimeout(250);
    report.mobile=await frame.evaluate(()=>{
      const c=document.querySelector('.model-stage canvas'),t=document.createElement('canvas');t.width=c.width;t.height=c.height;
      const ctx=t.getContext('2d');ctx.drawImage(c,0,0);const p=ctx.getImageData(0,0,t.width,t.height).data;
      let minX=t.width,maxX=-1,minY=t.height,maxY=-1,count=0;
      for(let y=0;y<t.height;y++)for(let x=0;x<t.width;x++){const i=(y*t.width+x)*4;if(p[i+3]>128&&Math.max(p[i],p[i+1],p[i+2])<170){minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);count++;}}
      return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,canvasWidth:t.width,canvasHeight:t.height,vehicleDarkPixelBounds:{minX,maxX,minY,maxY,count}};
    });
    report.mobile_outer=await page.evaluate(()=>({width:innerWidth,scrollWidth:document.documentElement.scrollWidth}));
    const b=report.mobile.vehicleDarkPixelBounds;
    report.mobile_vehicle_not_clipped=b.count>0&&b.minX>0&&b.maxX<report.mobile.canvasWidth-1&&b.minY>0&&b.maxY<report.mobile.canvasHeight-1;
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    await page.setViewportSize({width:800,height:1050});await page.emulateMedia({colorScheme:'dark'});await page.waitForTimeout(250);
    report.dark_theme=await frame.evaluate(()=>({theme:document.documentElement.dataset.theme,colorScheme:getComputedStyle(document.documentElement).colorScheme}));
    await page.screenshot({path:path.join(output,'dark.png'),fullPage:true});
    report.errors=errors;report.failed_requests=failed;
    report.passed=!errors.length&&!failed.length&&report.texture_switch_changes_pixels&&report.switch_retains_camera&&report.capture_size_correct&&report.roundtrip_original_pixels_exact&&report.drag_changes_view&&report.mobile.scrollWidth<=report.mobile.width&&report.mobile_outer.scrollWidth<=report.mobile_outer.width&&report.mobile_vehicle_not_clipped&&report.dark_theme.theme==='dark';
    fs.writeFileSync(path.join(output,'verification.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report,null,2));
    if(!report.passed)process.exitCode=1;
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
