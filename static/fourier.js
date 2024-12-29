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

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = (2 * Math.PI) / len; // Positive phase for DFT matching

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                const wReal = Math.cos(angle);
                const wImag = Math.sin(angle); // Positive sine matches DFT

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

function ifft(inp, ss) {
    const N = inp.length;

    // Ensure input size is a power of 2
    if ((N & (N - 1)) !== 0) {
        throw new Error("Input size must be a power of 2");
    }

    let s = ss * 1;

    // Extract real and imaginary parts
    const real = inp.map(x => math.re(x));
    const imag = inp.map(x => math.im(x));

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

    // Cooley-Tukey FFT
    for (let len = 2; len <= N; len *= 2) {
        const halfLen = len / 2;
        const angleStep = -(2 * Math.PI) / len; // Negative phase for IFFT

        for (let i = 0; i < N; i += len) {
            for (let j = 0; j < halfLen; j++) {
                const angle = angleStep * j;
                const wReal = Math.cos(angle);
                const wImag = Math.sin(angle); // Negative sine matches IFFT

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
