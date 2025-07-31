Python bindings for RustFFT, notably providing planned FFTs (meaning the bulk of the effort is cached and reusable)
without platform restrictions and with permissive licensing.

```python
import numpy as np
from rustfft import FftPlanner

# Complex-valued input is consumed as output buffer
n = 1024
buffer = np.random.uniform(0.0, 1.0, n).astype(np.complex128)

# Pre-planned FFT caches the initial setup
fft = FftPlanner(n)

# Run FFT repeatedly without re-initializing
fft.process(buffer)
fft.process(buffer)
```

# License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.