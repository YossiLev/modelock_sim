
var sfs = -1;
var fronts = [];
var ranges = [];
var locations = [];
var viewOption = 1;

function getInitMultyMode() {
    const sel = document.getElementById("incomingFront");
    let vf = [];
    let n = 512;
    console.log("Sel ", sel.value)
    switch (sel.value) {
        case "Gaussian Beam":
            z = n / 2 - 0.5;
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(1 * Math.exp(-(i - z) * (i - z) / 500)))
            }
            break;
        case "Two Slit":
            z1 = 2 * n / 5
            z2 = 3 * n / 5
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(1 * (Math.exp(-(i - z1) * (i - z1) / 50) + Math.exp(-(i - z2) * (i - z2) / 50))))
            }
            break;
        case "Delta":
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(0))
            }
            vf[n / 2] = math.complex(1);
            break;
        case "Zero":
            for (let i = 0; i < n; i++) {
                vf.push(math.complex(0))
            }
            break;    
    }

    // for (let i = 0; i < 20; i++) {
    //     vf[i] = math.complex(0);
    //     vf[n - 1 - i] = math.complex(0);
    // }

    console.log(vf);
    return vf;
}

function drawMultiMode() {
    const canvas = document.getElementById("funCanvas");
    const ctx = canvas.getContext("2d");
    const sx = 20, sy = 20;
    const w = 4;
    ctx.fillStyle = `red`;
    ctx.fillRect(0, 0, 1000, 1000);

    for (let f = 0; f < fronts.length; f++) {
        let fi = fronts[f];
        let r = ranges[f];
        let l = fi.length;       
        let h = r / l * 45000;
        for (let i = 0; i < l; i++) {
            if (viewOption == 1) {
                c = Math.floor(fi[i].toPolar().r * 255.0);
            } else {
                c = Math.floor((fi[i].toPolar().phi / (2 * Math.PI) + 0.5) * 255.0);
            }
            ctx.fillStyle = `rgba(${c}, ${c}, ${c}, 255)`;
            //console.log(`h = ${h} p = ${(i - (l / 2)) * h + 400} c  ${c}`)
            ctx.fillRect(sx + f * w, (i - (l / 2)) * (h) + 400, w, h + 1);
        }
    }
}

function initMultiMode() {
    fronts = [getInitMultyMode()];
    ranges = [0.001];
    sfs = 0;
    locations = [0];
    drawMultiMode();
}

/**
Discrete Fourier transform (DFT).
(the slowest possible implementation)
Assumes `inpReal` and `inpImag` arrays have the same size.
*/
function dft(inp, ss) {
    const out = [];
    const sin = [];
    const cos = [];
    let inpReal = [];
    let inpImag = [];
    let s = ss * 1;
  
    const N = inp.length;
    const twoPiByN = 2 * Math.PI / N;
  
    /* initialize Sin / Cos tables */
    for (let k = 0; k < N; k++) {
      inpReal.push(math.re(inp[k]));
      inpImag.push(math.im(inp[k]));
      const angle = twoPiByN * k;
      sin.push(Math.sin(angle));
      cos.push(Math.cos(angle));
    }
  
    for (let k = 0; k < N; k++) {
      let sumReal = 0;
      let sumImag = 0;
      let nn = 0;
      for (let n = 0; n < N; n++) {
        nm = (n + N / 2) % N;
        sumReal +=  inpReal[nm] * cos[nn] + inpImag[nm] * sin[nn];
        sumImag += -inpReal[nm] * sin[nn] + inpImag[nm] * cos[nn];
        nn = (nn + k) % N;
      }
      out.push(math.complex(sumReal * s, sumImag * s));
    }
    let o = [];
    for (let k = 0; k < N; k++) {
        o.push(out[(k + N / 2) % N]);
    }
    return o;
}

function propogateMultiMode() {
    if (fronts.length <= 0) {
        return;
    }
    let distS = 0.002;
    let lambda = 0.00000051;
    lfs = fronts.length;
    let dist = distS * (lfs - sfs);
    fi = math.clone(fronts[sfs]);
    let r = ranges[sfs];
    let L = fi.length;
    let dxi = r / L;
    let dxf = lambda * dist / r;
    // dxf = dxi;
    // lambda = range_i * dxf / dist;
    let factor = math.divide(math.exp(math.complex(0, dist * Math.PI * 2 / lambda)), math.complex(dist));
    let ff = Math.sqrt(1 / (dist * lambda * 2));
    factor = math.complex(- ff, ff);
    let coi = Math.PI * dxi * dxi / (dist * lambda);
    console.log(`factor = ${factor}, lambda = ${lambda}`)

    let cof = Math.PI * dxf * dxf / (dist * lambda);
    console.log(`dxi = ${dxi}, dxf = ${dxf}, coi = ${coi}, cof = ${cof}, r = ${r}, dist = ${dist}`)

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        fi[i] = math.multiply(fi[i], math.exp(math.complex(0, coi * ii * ii)))
    }
    ff = dft(fi, dxi);

    for (let i = 0; i < L; i++) {
        let ii = i - L / 2;
        ff[i] = math.multiply(math.multiply(ff[i], factor), math.exp(math.complex(0, cof * ii * ii)))
    }

    fronts.push(ff);
    ranges.push(L * dxf);

    drawMultiMode();
}

function lensMultiMode() {
    if (fronts.length <= 0) {
        return;
    }

    fl = fronts.length
    ff = math.clone(fronts[fl - 1]);
    let r = ranges[fl - 1];
    let L = fi.length;
    let dx = r / L;

    let z = L / 2.0;
    for (let i = 0; i < L; i++) {
        let factor = (i - z) * (i - z) * dx * dx
        console.log(`i = ${i}, factor = ${factor}`);
        ff[i] = math.multiply(ff[i], math.exp(math.complex(- factor * 10000,  - factor * 100000000)))
    }

    sfs = fl;
    fronts.push(ff);
    ranges.push(ranges[sfs - 1]);
    console.log(`LENS range = ${ranges[sfs]}`)
    drawMultiMode();
}

function switchViewMultiMode() {
    viewOption = 1 - viewOption;
    drawMultiMode();
}