
var fronts = [];
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
    const w = 4, h = 1;
    for (let f = 0; f < fronts.length; f++) {
        for (let i = 0; i < fronts[f].length; i++) {
            if (viewOption == 1) {
                c = Math.floor(fronts[f][i].toPolar().r * 255.0);
            } else {
                c = Math.floor((fronts[f][i].toPolar().phi / (2 * Math.PI) + 0.5) * 255.0);
            }
            ctx.fillStyle = `rgba(${c}, ${c}, ${c}, 255)`;
            ctx.fillRect(sx + f * (w + 0), sy + i * h, w, h);
        }
    }
}

function initMultiMode() {
    fronts = [getInitMultyMode()];
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
        sumReal +=  inpReal[n] * cos[nn] + inpImag[n] * sin[nn];
        sumImag += -inpReal[n] * sin[nn] + inpImag[n] * cos[nn];
        nn = (nn + k) % N;
      }
      out.push(math.complex(sumReal * s, sumImag * s));
    }
    return out;
};

function propogateMultiMode() {
    if (fronts.length <= 0) {
        return;
    }
    let range_i = 0.0035;
    let dist = 0.005;
    let lambda = 0.000001;

    fi = math.clone(fronts[fronts.length - 1]);
    let L = fi.length;
    let dxi = range_i / L;
    let dxf = lambda * dist / range_i;
    // dxf = dxi;
    // lambda = range_i * dxf / dist;
    let factor = math.divide(math.exp(math.complex(0, dist * Math.PI * 2 / lambda)), math.complex(dist));
    let ff = Math.sqrt(1 / (dist * lambda * 2));
    factor = math.complex(- ff, ff);
    let coi = Math.PI * dxi * dxi / (dist * lambda);
    console.log(`factor = ${factor}, lambda = ${lambda}`)

    let cof = Math.PI * dxf * dxf / (dist * lambda);
    console.log(`dxi = ${dxi}, dxf = ${dxf}, coi = ${coi}, cof = ${cof}, `)

    for (let i = 0; i < L; i++) {
        fi[i] = math.multiply(fi[i], math.exp(math.complex(0, coi * i * i)))
    }
    ff = dft(fi, dxi);

    for (let i = 0; i < L; i++) {
        ff[i] = math.multiply(math.multiply(ff[i], factor), math.exp(math.complex(0, cof * i * i)))
    }

    fronts.push(ff);
    drawMultiMode();
}

function lensMultiMode() {
    if (fronts.length <= 0) {
        return;
    }

    ff = math.clone(fronts[fronts.length - 1]);
    let L = fi.length;

    let z = L / 2.0;
    for (let i = 0; i < L; i++) {
        ff[i] = math.multiply(ff[i], math.exp(math.complex(0, - (i - z) * (i - z) / 10000)))
    }

    fronts.push(ff);
    drawMultiMode();
}

function switchViewMultiMode() {
    viewOption = 1 - viewOption;
    drawMultiMode();
}