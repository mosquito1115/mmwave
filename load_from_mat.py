import numpy as np

def _try_load_mat(mat_path: str):
    """
    Load a MAT file, supporting both v7 (scipy.io.loadmat) and v7.3 (HDF5 via h5py).

    Returns a dict-like object mapping variable names to numpy arrays or h5py datasets.
    """
    # Try scipy first
    try:
        from scipy.io import loadmat  # type: ignore
        md = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        # Remove meta keys that scipy adds
        return {k: v for k, v in md.items() if not k.startswith('__')}
    except Exception as e_scipy:
        # Fall back to h5py for v7.3
        try:
            import h5py  # type: ignore
            f = h5py.File(mat_path, 'r')
            return f
        except Exception as e_h5:
            raise RuntimeError(
                f"Failed to load MAT file with scipy and h5py. scipy error: {e_scipy}; h5py error: {e_h5}"
            )

def _as_numpy(arr):
    """
    Convert scipy or h5py-stored arrays to numpy arrays with native dtype.
    """
    try:
        import numpy as _np
        if 'h5py' in type(arr).__module__:
            # h5py datasets need [:] to realize
            return _np.array(arr[...])
        return _np.array(arr)
    except Exception:
        return np.array(arr)

def _load_signal_and_dims(store, var_name: str = 'signal_raw'):
    """
    Extract `signal_raw` and optionally `numLoops` / `numFrames` if present.
    Returns tuple (signal_raw_np, numLoops, numFrames) where latter may be inferred.
    """
    # Access strategy differs for scipy dict vs h5py group
    is_h5 = hasattr(store, 'keys') and not isinstance(store, dict)
    if isinstance(store, dict):
        if var_name not in store:
            raise KeyError(f"Variable {var_name} not found in MAT file")
        sig = _as_numpy(store[var_name])
        return sig
    else:
        # h5py File or Group
        if var_name not in store:
            # Try common nesting patterns
            candidates = [k for k in store.keys() if var_name in k]
            if candidates:
                key = candidates[0]
            else:
                raise KeyError(f"Variable {var_name} not found in MAT file (h5)")
        else:
            key = var_name
        sig = _as_numpy(store[key])
        return sig

def load_from_mat(
    mat_path: str,
    antenna_idx: int = 0,
    sample_idx: int = 19,
    loop_start: int = 1,
    angle_idx: int = 0,
    var_name: str = 'signal_raw',
):
    """
    Mirror MATLAB:
        temp_data = squeeze(signal_raw(1,20,2:end,1,:));
        data = reshape(temp_data, [], (numLoops-1)*numFrames);

    In Python (0-based): antenna_idx=0, sample_idx=19, loops 1:, angle_idx=0, frames :
    Returns: 1-D complex ndarray of length (numLoops-1)*numFrames (row-major equivalent).
    """
    store = _try_load_mat(mat_path)
    sig = _load_signal_and_dims(store, var_name=var_name)

    numFrames, numAngles, numLoops, numSamples, numAntennas = sig.shape

    # Slice per MATLAB instruction
    sel = sig[:, 0, loop_start:, sample_idx, antenna_idx]

    # Ensure complex dtype if stored as real-imag parts along last dim; otherwise cast
    sel_np = np.asarray(sel)
    mat2d = np.squeeze(sel_np['real'] + 1j * sel_np['imag'])

    Fnum, Lnum = mat2d.shape

    # Reshape to 1 x ((numLoops-1)*numFrames) in MATLAB (column-major).
    # In numpy (row-major), flatten in column-major to match MATLAB behavior.
    vec = np.reshape(mat2d, (Lnum * Fnum,), order='C')
    return vec.astype(np.complex128), int(numLoops), int(numFrames)