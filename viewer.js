import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

const W=512,H=512,canvas=document.getElementById('c');
const status=document.getElementById('status'),diagnostic=document.getElementById('diagnostics');
const config={eye:[4.8,3.8,6.7],target:[0,1.5,0],fovy:40,near:1,far:50,ambient:.15,light:[.4,.9,.6]};
let renderer,scene,camera,controls,model,texture,previousTime;
const renderTimes=[],frameIntervals=[];
function download(name,blob){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function resetCamera(){camera.position.fromArray(config.eye);controls.target.fromArray(config.target);camera.up.set(0,1,0);controls.update();}
function setTextureMode(){const smooth=document.getElementById('texture').value==='smooth';texture.generateMipmaps=smooth;texture.minFilter=smooth?THREE.LinearMipmapLinearFilter:THREE.LinearFilter;texture.magFilter=THREE.LinearFilter;texture.needsUpdate=true;}
function metadata(){
  model.updateMatrixWorld(true);camera.updateMatrixWorld(true);
  const median=a=>a.length?[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)]:null;
  const gl=renderer.getContext();
  return {capturedAt:new Date().toISOString(),userAgent:navigator.userAgent,threeRevision:THREE.REVISION,ready:true,
    triangles:renderer.info.render.triangles,drawingBuffer:[gl.drawingBufferWidth,gl.drawingBufferHeight],pixelRatio:renderer.getPixelRatio(),
    antialias:gl.getContextAttributes().antialias,webgl:gl.getParameter(gl.VERSION),outputColorSpace:renderer.outputColorSpace,
    textureMode:document.getElementById('texture').value,wireframe:document.getElementById('wire').getAttribute('aria-pressed')==='true',
    camera:{position:camera.position.toArray(),target:controls.target.toArray(),fovy:camera.fov,near:camera.near,far:camera.far},
    modelMatrix:model.matrixWorld.toArray(),viewMatrix:camera.matrixWorldInverse.toArray(),projectionMatrix:camera.projectionMatrix.toArray(),
    cpuRenderSubmissionMedianMs:median(renderTimes),rafIntervalMedianMs:median(frameIntervals),timingSamples:renderTimes.length,lighting:config};
}
function exportPPM(){
  renderer.render(scene,camera);
  const gl=renderer.getContext(),rgba=new Uint8Array(W*H*4),rgb=new Uint8Array(W*H*3);
  gl.readPixels(0,0,W,H,gl.RGBA,gl.UNSIGNED_BYTE,rgba);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){const s=((H-1-y)*W+x)*4,d=(y*W+x)*3;rgb[d]=rgba[s];rgb[d+1]=rgba[s+1];rgb[d+2]=rgba[s+2];}
  download('mymodel_three.ppm',new Blob([new TextEncoder().encode(`P6\n${W} ${H}\n255\n`),rgb],{type:'application/octet-stream'}));
  status.textContent='Frame 512 × 512 diekspor';
}
async function start(){
  renderer=new THREE.WebGLRenderer({canvas,antialias:true,preserveDrawingBuffer:true,alpha:false});
  renderer.setPixelRatio(1);renderer.setSize(W,H,false);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
  scene=new THREE.Scene();scene.background=new THREE.Color(0);
  camera=new THREE.PerspectiveCamera(config.fovy,W/H,config.near,config.far);
  controls=new OrbitControls(camera,canvas);controls.enableDamping=false;controls.minDistance=2;controls.maxDistance=35;resetCamera();
  // r160 Lambert divides irradiance by pi; compensate to match NumPy .15+.85*N.L.
  scene.add(new THREE.AmbientLight(0xffffff,Math.PI*config.ambient));
  const sun=new THREE.DirectionalLight(0xffffff,Math.PI*(1-config.ambient));sun.position.fromArray(config.light);scene.add(sun);
  [model,texture]=await Promise.all([new OBJLoader().loadAsync('./assets/mymodel.obj'),new THREE.TextureLoader().loadAsync('./assets/lab1_brick.png')]);
  texture.colorSpace=THREE.SRGBColorSpace;texture.wrapS=texture.wrapT=THREE.RepeatWrapping;texture.flipY=true;
  // Match CPU u*(size-1) with GPU texel-center coordinates. Actual UVs lie in .02.. .98.
  texture.repeat.set((texture.image.width-1)/texture.image.width,(texture.image.height-1)/texture.image.height);
  texture.offset.set(.5/texture.image.width,.5/texture.image.height);setTextureMode();
  const material=new THREE.MeshLambertMaterial({map:texture,side:THREE.DoubleSide,flatShading:true});
  let triangles=0;model.traverse(c=>{if(c.isMesh){c.material=material;triangles+=c.geometry.attributes.position.count/3;}});scene.add(model);
  if(triangles!==636)throw new Error(`Jumlah triangle tidak sesuai: ${triangles}`);
  for(const id of ['reset','wire','texture','export','verify'])document.getElementById(id).disabled=false;
  document.getElementById('reset').addEventListener('click',()=>{resetCamera();status.textContent='Tampilan awal dipulihkan';});
  document.getElementById('wire').addEventListener('click',e=>{const on=e.currentTarget.getAttribute('aria-pressed')!=='true';model.traverse(c=>{if(c.isMesh)c.material.wireframe=on;});e.currentTarget.setAttribute('aria-pressed',String(on));status.textContent=on?'Kerangka model aktif':'Permukaan model aktif';});
  document.getElementById('texture').addEventListener('change',()=>{setTextureMode();status.textContent='Tampilan permukaan diperbarui';});
  document.getElementById('export').addEventListener('click',exportPPM);
  document.getElementById('verify').addEventListener('click',()=>download('mymodel_verification.json',new Blob([JSON.stringify(metadata(),null,2)],{type:'application/json'})));
  canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();status.textContent='Konteks WebGL terhenti. Muat ulang halaman.';status.classList.add('error');});
  status.textContent='636 segitiga · siap dijelajahi';let frame=0;
  function animate(time){requestAnimationFrame(animate);controls.update();const before=performance.now();renderer.render(scene,camera);renderTimes.push(performance.now()-before);if(previousTime!==undefined)frameIntervals.push(time-previousTime);previousTime=time;if(renderTimes.length>240)renderTimes.shift();if(frameIntervals.length>240)frameIntervals.shift();if(frame++%60===0)diagnostic.textContent=JSON.stringify(metadata(),null,2);}
  requestAnimationFrame(animate);
}
start().catch(e=>{status.textContent=`Gagal memuat viewer: ${e.message}`;status.classList.add('error');diagnostic.textContent=e.stack||String(e);console.error(e);});
