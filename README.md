Python bindings for RustFFT, notably providing planned FFTs (meaning the bulk of the effort is cached and reusable)
without platform restrictions and with permissive licensing.

```python
import numpy as np
from rustfft import FftPlanner

for direction in ["forward", "inverse"]:
    # Supports complex numbers with 32 and 64-bit floats
    for dtype in [np.complex64, np.complex128]:
        # Complex-valued input
        n = 1024
        buffer = np.ascontiguousarray(np.random.uniform(0.0, 1.0, n).astype(dtype))
        original = buffer.copy()

        # Pre-planned FFT caches the initial setup
        fft = FftPlanner(n, direction, dtype)

        # Run FFT, reusing input for output storage to avoid allocation if possible
        out = fft.process(buffer)
        if direction == "forward":
            assert np.allclose(out, np.fft.fft(original))
        else:
            assert np.allclose(out, np.fft.ifft(original))

        # Run FFT repeatedly without re-initializing
        out = fft.process(out)
```

# License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.