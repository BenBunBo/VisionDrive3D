"""
Async framebuffer readback via Pixel Buffer Objects (PBO).

Why: a plain ``glReadPixels`` blocks the CPU until the GPU has finished all
pending rendering (a full pipeline flush). When exporting datasets we used to
do this after *every* render pass (21 stalls per exported frame).

With PBOs, ``glReadPixels`` into a bound ``GL_PIXEL_PACK_BUFFER`` only queues
an asynchronous DMA transfer and returns immediately. We can therefore:

1. render pass -> enqueue readback (returns instantly, GPU keeps working)
2. ... repeat for all passes/cameras ...
3. call :meth:`PBOReader.fetch_all` -> one single ``glFinish`` for the whole
   frame, then map every PBO to a numpy array.

Usage:
    reader = PBOReader()
    i_rgb = reader.enqueue_rgb(w, h)
    # ... more passes / readbacks ...
    frames = reader.fetch_all()   # single sync point
    rgb = frames[i_rgb]           # (h, w, 3) uint8, already flipud'ed
"""
import ctypes

import numpy as np
import OpenGL.GL as GL


class PBOReader:
    """Queue async glReadPixels into PBOs; sync once and fetch everything."""

    _NP_TO_GL_TYPE = {
        np.dtype(np.uint8): GL.GL_UNSIGNED_BYTE,
        np.dtype(np.float32): GL.GL_FLOAT,
    }

    def __init__(self):
        # (pbo_id, width, height, channels, np_dtype, nbytes)
        self._entries = []
        self._prev_pack_alignment = None

    # ------------------------------------------------------------------
    # Enqueue (non-blocking)
    # ------------------------------------------------------------------
    def enqueue(self, width, height, gl_format, np_dtype, channels) -> int:
        """
        Queue an async readback of the current read framebuffer.
        Returns the index to use with the list from fetch_all().
        """
        width, height = int(width), int(height)
        np_dtype = np.dtype(np_dtype)
        gl_type = self._NP_TO_GL_TYPE[np_dtype]
        nbytes = width * height * channels * np_dtype.itemsize

        # Tight packing so the PBO size is exactly w*h*c*itemsize.
        if self._prev_pack_alignment is None:
            self._prev_pack_alignment = int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT))
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)

        pbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
        GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, nbytes, None, GL.GL_STREAM_READ)
        # NULL data pointer + bound PBO => async read into PBO offset 0.
        GL.glReadPixels(0, 0, width, height, gl_format, gl_type, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

        self._entries.append((pbo, width, height, channels, np_dtype, nbytes))
        return len(self._entries) - 1

    def enqueue_rgb(self, width, height) -> int:
        """Read the color buffer as (h, w, 3) uint8."""
        return self.enqueue(width, height, GL.GL_RGB, np.uint8, 3)

    def enqueue_depth_float(self, width, height) -> int:
        """Read the depth buffer as (h, w) float32 in [0, 1]."""
        return self.enqueue(width, height, GL.GL_DEPTH_COMPONENT, np.float32, 1)

    # ------------------------------------------------------------------
    # Fetch (single synchronization point)
    # ------------------------------------------------------------------
    def fetch_all(self) -> list:
        """
        Wait once for the GPU to finish everything queued so far, then map
        all PBOs and return the images as a list of numpy arrays (in enqueue
        order). Arrays are flipped vertically (GL origin is bottom-left).
        """
        GL.glFinish()  # THE one and only CPU<->GPU sync for this frame

        results = []
        for pbo, w, h, channels, np_dtype, nbytes in self._entries:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
            ptr = GL.glMapBuffer(GL.GL_PIXEL_PACK_BUFFER, GL.GL_READ_ONLY)
            try:
                if not ptr:
                    raise RuntimeError("glMapBuffer failed on PBO readback")
                raw = ctypes.string_at(ptr, nbytes)
                arr = np.frombuffer(raw, dtype=np_dtype, count=w * h * channels)
                arr = arr.reshape(h, w, channels) if channels > 1 else arr.reshape(h, w)
                results.append(np.flipud(arr).copy())
            finally:
                GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
                GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
                GL.glDeleteBuffers(1, [pbo])

        self._entries.clear()
        if self._prev_pack_alignment is not None:
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, self._prev_pack_alignment)
            self._prev_pack_alignment = None
        return results
