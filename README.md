Python bindings for RustFFT, notably providing planned FFTs (meaning the bulk of the effort is cached and reusable)
without platform restrictions and with permissive licensing.

```python
import numpy as np
from rustfft import FftPlanner

rng = np.random.default_rng(seed=89324598)

# Supports complex numbers with 32 and 64-bit floats
n = 1024
for dtype in [np.complex64, np.complex128]:
    # Complex-valued input
    buffer = np.ascontiguousarray(rng.uniform(0.0, 1.0, n).astype(dtype))  # Real input
    original = buffer.copy()

    # Pre-planned FFT caches the initial setup
    fft = FftPlanner(n, dtype=dtype)

    # Run FFT, reusing input for output storage to avoid allocation if possible
    out = fft.process(buffer)
    assert np.allclose(out, np.fft.fft(original)), "Forward results should match numpy"

    # Inverse supported
    ifft = FftPlanner(n, "inverse", dtype)
    assert np.allclose(ifft(out), original), "`ifft(fft(x))` should restore `x`"

    # Run FFT repeatedly without re-initializing
    out = fft.process(out)
```

# License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.