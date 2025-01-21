// thank you chatgpt
function test() {
    let inp = [
        math.complex(2, 0),
        math.complex(7, 0),
        math.complex(-1, 0),
        math.complex(0, 0),
        math.complex(1, 0),
        math.complex(0, 0),
        math.complex(-1, 0),
        math.complex(0, 0),
    ];

    let inp2 = math.clone(inp);

    const ss = 1; // Scaling factor
    const fftOutput = fft(inp, ss);
    console.log("FFT Output:", fftOutput);

    const dftOutput = dft(inp2, ss);
    console.log("DFT Output:", dftOutput);

}

function fft(inp, s) {
    const N = inp.length;

    // Ensure input size is a power of 2
    if ((N & (N - 1)) !== 0) {
        throw new Error("Input size must be a power of 2");
    }

    // Extract real and imaginary parts
    const realS = inp.map(x => math.re(x));
    const imagS = inp.map(x => math.im(x));

    // Reorder input
    let real = [], imag = [];
    for (let k = 0; k < N; k++) {
        real.push(realS[(k + N / 2) % N]);
        imag.push(imagS[(k + N / 2) % N]);
    }

    // Bit-reversal permutation
    function bitReverseArray(arr) {
        const n = arr.length;
        const reversed = new Array(n);
        let bits = Math.log2(n);

        for (let i = 0; i < n; i++) {
            let reversedIndex = 0;
            for (let j = 0; j < bits; j++) {
                reversedIndex = (reversedIndex << 1) | ((i >> j) & 1);
            }
            reversed[reversedIndex] = arr[i];
        }

        return reversed;
    }

    const realReversed = bitReverseArray(real);
    const imagReversed = bitReverseArray(imag);

    //const sinVec = realReversed.map((x, i) => Math.sin(s * Math.PI * i / N));
    //const cosVec = realReversed.map((x, i) => Math.cos(s * Math.PI * i / N));

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = (2 * Math.PI) / len; // Positive phase for DFT matching
        //const angleStepI = 2 * N / len;

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                //const angleI = angleStepI * j; 
                const wReal = Math.cos(angle);
                const wImag = Math.sin(angle); // Positive sine matches DFT
                //const wReal = cosVec[angleI];
                //const wImag = sinVec[angleI];
                // if (Math.abs(wReal - wRealI) < 0.0000001 && Math.abs(wImag - wImagI) < 0.0000001) {
                //     console.log("OK");
                // } else {
                //     console.log(`BAD j = ${j} angle = cos(${angle}) = ${Math.cos(angle)} angleI = veccos(${angleI}) = ${cosVec[angleI]}`);
                //     console.log(`BAD j = ${j} angle = sin(${angle}) = ${Math.sin(angle)} angleI = vecsin(${angleI}) = ${sinVec[angleI]}`);
                // }

                const evenReal = realReversed[i + j];
                const evenImag = imagReversed[i + j];
                const oddReal = realReversed[i + j + halfLen];
                const oddImag = imagReversed[i + j + halfLen];

                // FFT butterfly computations (positive sine matches DFT)
                const tReal = wReal * oddReal - wImag * oddImag;
                const tImag = wReal * oddImag + wImag * oddReal;

                realReversed[i + j] = evenReal + tReal;
                imagReversed[i + j] = evenImag + tImag;

                realReversed[i + j + halfLen] = evenReal - tReal;
                imagReversed[i + j + halfLen] = evenImag - tImag;
            }
        }
    }

    // Scale results while reordering 
    const output = [];
    for (let i = 0; i < N; i++) {
        let iFrom = (i + N / 2) % N;
        output.push(math.complex(realReversed[iFrom] * s, imagReversed[iFrom] * s));
    }

    return output;
}

function ifft(inp, s) {
    const N = inp.length;

    // Ensure input size is a power of 2
    if ((N & (N - 1)) !== 0) {
        throw new Error("Input size must be a power of 2");
    }

    // Extract real and imaginary parts
    const realS = inp.map(x => math.re(x));
    const imagS = inp.map(x => math.im(x));

    // Reorder input
    let real = [], imag = [];
    for (let k = 0; k < N; k++) {
        real.push(realS[(k + N / 2) % N]);
        imag.push(imagS[(k + N / 2) % N]);
    }


    // Bit-reversal permutation
    function bitReverseArray(arr) {
        const n = arr.length;
        const reversed = new Array(n);
        let bits = Math.log2(n);

        for (let i = 0; i < n; i++) {
            let reversedIndex = 0;
            for (let j = 0; j < bits; j++) {
                reversedIndex = (reversedIndex << 1) | ((i >> j) & 1);
            }
            reversed[reversedIndex] = arr[i];
        }

        return reversed;
    }

    const realReversed = bitReverseArray(real);
    const imagReversed = bitReverseArray(imag);

    //const sinVec = realReversed.map((x, i) => Math.sin(s * Math.PI * i / N));
    //const cosVec = realReversed.map((x, i) => Math.cos(s * Math.PI * i / N));

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = -(2 * Math.PI) / len; // Negative phase for IFFT
        //const angleStepI = 2 * N / len;

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                //const angleI = angleStepI * j;
                const wReal = Math.cos(angle);
                const wImag = Math.sin(angle); // Negative sine matches IFFT
                //const wReal = cosVec[angleI];
                //const wImag = - sinVec[angleI];
 
                const evenReal = realReversed[i + j];
                const evenImag = imagReversed[i + j];
                const oddReal = realReversed[i + j + halfLen];
                const oddImag = imagReversed[i + j + halfLen];

                // FFT butterfly computations (negative sine for IFFT)
                const tReal = wReal * oddReal - wImag * oddImag;
                const tImag = wReal * oddImag + wImag * oddReal;

                realReversed[i + j] = evenReal + tReal;
                imagReversed[i + j] = evenImag + tImag;

                realReversed[i + j + halfLen] = evenReal - tReal;
                imagReversed[i + j + halfLen] = evenImag - tImag;
            }
        }
    }

    // Scale results
    const output = [];
    for (let i = 0; i < N; i++) {
        output.push(math.complex((realReversed[i] * s) / N, (imagReversed[i] * s) / N));
    }

    // Reorder output as in the DFT
    let reorderedOutput = [];
    for (let k = 0; k < N; k++) {
        reorderedOutput.push(output[(k + N / 2) % N]);
    }

    return reorderedOutput;
}


function fftx(inp) {
    const N = inp.length;
    const Nd2 = N / 2;

    let reorder = inp.map((x, i, a) => i < Nd2 ? a[i + Nd2] : a[i - Nd2]);
    // // Extract real and imaginary parts
    // const realS = inp.map(x => math.re(x));
    // const imagS = inp.map(x => math.im(x));

    // // Reorder input
    // let real = [], imag = [];
    // for (let k = 0; k < N; k++) {
    //     real.push(realS[(k + N / 2) % N]);
    //     imag.push(imagS[(k + N / 2) % N]);
    // }

    // Bit-reversal permutation
    function bitReverseArray(arr) {
        const n = arr.length;
        const reversed = new Array(n);
        let bits = Math.log2(n);

        for (let i = 0; i < n; i++) {
            let reversedIndex = 0;
            for (let j = 0; j < bits; j++) {
                reversedIndex = (reversedIndex << 1) | ((i >> j) & 1);
            }
            reversed[reversedIndex] = arr[i];
        }

        return reversed;
    }

    const reversed = bitReverseArray(reorder);
    // const realReversed = bitReverseArray(real);
    // const imagReversed = bitReverseArray(imag);

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = (2 * Math.PI) / len; // Positive phase for DFT matching

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                const w = math.exp(math.complex(0, angle));
                // const wReal = Math.cos(angle);
                // const wImag = Math.sin(angle); // Positive sine matches DFT

                const even = reversed[i + j];
                const odd = reversed[i + j + halfLen];
                // const evenReal = realReversed[i + j];
                // const evenImag = imagReversed[i + j];
                // const oddReal = realReversed[i + j + halfLen];
                // const oddImag = imagReversed[i + j + halfLen];

                // FFT butterfly computations (positive sine matches DFT)
                const t = math.multiply(w, odd);
                // const tReal = wReal * oddReal - wImag * oddImag;
                // const tImag = wReal * oddImag + wImag * oddReal;

                reversed[i + j] = math.add(even, t);
                // realReversed[i + j] = evenReal + tReal;
                // imagReversed[i + j] = evenImag + tImag;

                reversed[i + j + halfLen] = math.subtract(even, t);
                // realReversed[i + j + halfLen] = evenReal - tReal;
                // imagReversed[i + j + halfLen] = evenImag - tImag;
            }
        }
    }

    const output = reversed.map((x, i, a) => i < Nd2 ? 
        math.divide(a[i + Nd2], N) : math.divide(a[i - Nd2], N));
    // Scale results while reordering 
    // const output = [];
    // for (let i = 0; i < N; i++) {
    //     let iFrom = (i + N / 2) % N;
    //     output.push(math.complex(realReversed[iFrom], imagReversed[iFrom]));
    // }

    return output;
}

function ifftx(inp) {
    const N = inp.length;
    const Nd2 = N / 2;

    // Ensure input size is a power of 2
    if ((N & (N - 1)) !== 0) {
        throw new Error("Input size must be a power of 2");
    }

    let reorder = inp.map((x, i, a) => i < Nd2 ? a[i + Nd2] : a[i - Nd2]);
    // // Extract real and imaginary parts
    // const realS = inp.map(x => math.re(x));
    // const imagS = inp.map(x => math.im(x));

    // // Reorder input
    // let real = [], imag = [];
    // for (let k = 0; k < N; k++) {
    //     real.push(realS[(k + N / 2) % N]);
    //     imag.push(imagS[(k + N / 2) % N]);
    // }

    // Bit-reversal permutation
    function bitReverseArray(arr) {
        const n = arr.length;
        const reversed = new Array(n);
        let bits = Math.log2(n);

        for (let i = 0; i < n; i++) {
            let reversedIndex = 0;
            for (let j = 0; j < bits; j++) {
                reversedIndex = (reversedIndex << 1) | ((i >> j) & 1);
            }
            reversed[reversedIndex] = arr[i];
        }

        return reversed;
    }

    const reversed = bitReverseArray(reorder);
    // const realReversed = bitReverseArray(real);
    // const imagReversed = bitReverseArray(imag);

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = -(2 * Math.PI) / len; // Negative phase for IFFT

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                const w = math.exp(math.complex(0, angle));
                // const wReal = Math.cos(angle);
                // const wImag = Math.sin(angle); // Positive sine matches DFT

                const even = reversed[i + j];
                const odd = reversed[i + j + halfLen];
                // const evenReal = realReversed[i + j];
                // const evenImag = imagReversed[i + j];
                // const oddReal = realReversed[i + j + halfLen];
                // const oddImag = imagReversed[i + j + halfLen];

                // FFT butterfly computations (positive sine matches DFT)
                const t = math.multiply(w, odd);
                // const tReal = wReal * oddReal - wImag * oddImag;
                // const tImag = wReal * oddImag + wImag * oddReal;

                reversed[i + j] = math.add(even, t);
                // realReversed[i + j] = evenReal + tReal;
                // imagReversed[i + j] = evenImag + tImag;

                reversed[i + j + halfLen] = math.subtract(even, t);
                // realReversed[i + j + halfLen] = evenReal - tReal;
                // imagReversed[i + j + halfLen] = evenImag - tImag;
            }
        }
    }

    const output = reversed.map((x, i, a) => i < Nd2 ? 
        math.divide(a[i + Nd2], N) : math.divide(a[i - Nd2], N));
    // Scale results while reordering 
    // const output = [];
    // for (let i = 0; i < N; i++) {
    //     let iFrom = (i + N / 2) % N;
    //     output.push(math.complex(realReversed[iFrom], imagReversed[iFrom]));
    // }

    return output;
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
  
    console.log("minus");
    /* initialize Sin / Cos tables */
    for (let k = 0; k < N; k++) {
      inpReal.push(math.re(inp[k]));
      inpImag.push(math.im(inp[k]));
      const angle = - twoPiByN * k;
      sin.push(Math.sin(angle));
      cos.push(Math.cos(angle));
    }
  
    for (let k = 0; k < N; k++) {
      let sumReal = 0;
      let sumImag = 0;
      let nn = 0;
      for (let iN = 0; iN < N; iN++) {
        nm = (iN + N / 2) % N;
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

