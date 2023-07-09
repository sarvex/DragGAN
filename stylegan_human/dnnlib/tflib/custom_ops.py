# Copyright (c) SenseTime Research. All rights reserved.

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""TensorFlow custom ops builder.
"""

import os
import re
import uuid
import hashlib
import tempfile
import shutil
import tensorflow as tf
from tensorflow.python.client import device_lib # pylint: disable=no-name-in-module

#----------------------------------------------------------------------------
# Global options.

cuda_cache_path = os.path.join(os.path.dirname(__file__), '_cudacache')
cuda_cache_version_tag = 'v1'
do_not_hash_included_headers = False # Speed up compilation by assuming that headers included by the CUDA code never change. Unsafe!
verbose = True # Print status messages to stdout.

compiler_bindir_search_path = [
    'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.14.26428/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.23.28105/bin/Hostx64/x64',
    'C:/Program Files (x86)/Microsoft Visual Studio 14.0/vc/bin',
]

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    return next(
        (
            compiler_path
            for compiler_path in compiler_bindir_search_path
            if os.path.isdir(compiler_path)
        ),
        None,
    )

def _get_compute_cap(device):
    caps_str = device.physical_device_desc
    m = re.search('compute capability: (\\d+).(\\d+)', caps_str)
    major = m[1]
    minor = m[2]
    return (major, minor)

def _get_cuda_gpu_arch_string():
    gpus = [x for x in device_lib.list_local_devices() if x.device_type == 'GPU']
    if not gpus:
        raise RuntimeError('No GPU devices found')
    (major, minor) = _get_compute_cap(gpus[0])
    return f'sm_{major}{minor}'

def _run_cmd(cmd):
    with os.popen(cmd) as pipe:
        output = pipe.read()
        status = pipe.close()
    if status is not None:
        raise RuntimeError('NVCC returned an error. See below for full command line and output log:\n\n%s\n\n%s' % (cmd, output))

def _prepare_nvcc_cli(opts):
    cmd = f'nvcc {opts.strip()}'
    cmd += ' --disable-warnings'
    cmd += f' --include-path "{tf.sysconfig.get_include()}"'
    cmd += f""" --include-path "{os.path.join(tf.sysconfig.get_include(), 'external', 'protobuf_archive', 'src')}\""""
    cmd += f""" --include-path "{os.path.join(tf.sysconfig.get_include(), 'external', 'com_google_absl')}\""""
    cmd += f""" --include-path "{os.path.join(tf.sysconfig.get_include(), 'external', 'eigen_archive')}\""""

    compiler_bindir = _find_compiler_bindir()
    if compiler_bindir is None:
        # Require that _find_compiler_bindir succeeds on Windows.  Allow
        # nvcc to use whatever is the default on Linux.
        if os.name == 'nt':
            raise RuntimeError(
                f'Could not find MSVC/GCC/CLANG installation on this computer. Check compiler_bindir_search_path list in "{__file__}".'
            )
    else:
        cmd += f' --compiler-bindir "{compiler_bindir}"'
    cmd += ' 2>&1'
    return cmd

#----------------------------------------------------------------------------
# Main entry point.

_plugin_cache = {}

def get_plugin(cuda_file):
    cuda_file_base = os.path.basename(cuda_file)
    cuda_file_name, cuda_file_ext = os.path.splitext(cuda_file_base)

    # Already in cache?
    if cuda_file in _plugin_cache:
        return _plugin_cache[cuda_file]

    # Setup plugin.
    if verbose:
        print(f'Setting up TensorFlow plugin "{cuda_file_base}": ', end='', flush=True)
    try:
        # Hash CUDA source.
        md5 = hashlib.md5()
        with open(cuda_file, 'rb') as f:
            md5.update(f.read())
        md5.update(b'\n')

        # Hash headers included by the CUDA code by running it through the preprocessor.
        if not do_not_hash_included_headers:
            if verbose:
                print('Preprocessing... ', end='', flush=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, f'{cuda_file_name}_tmp{cuda_file_ext}')
                _run_cmd(
                    _prepare_nvcc_cli(
                        f'"{cuda_file}" --preprocess -o "{tmp_file}" --keep --keep-dir "{tmp_dir}"'
                    )
                )
                with open(tmp_file, 'rb') as f:
                    bad_file_str = ('"' + cuda_file.replace('\\', '/') + '"').encode('utf-8') # __FILE__ in error check macros
                    good_file_str = f'"{cuda_file_base}"'.encode('utf-8')
                    for ln in f:
                        if not ln.startswith(b'# ') and not ln.startswith(b'#line '): # ignore line number pragmas
                            ln = ln.replace(bad_file_str, good_file_str)
                            md5.update(ln)
                    md5.update(b'\n')

        # Select compiler options.
        compile_opts = ''
        if os.name == 'nt':
            compile_opts += f""""{os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.lib')}\""""
        elif os.name == 'posix':
            compile_opts += f""""{os.path.join(tf.sysconfig.get_lib(), 'python', '_pywrap_tensorflow_internal.so')}\""""
            compile_opts += ' --compiler-options \'-fPIC -D_GLIBCXX_USE_CXX11_ABI=0\''
        else:
            assert False # not Windows or Linux, w00t?
        compile_opts += f' --gpu-architecture={_get_cuda_gpu_arch_string()}'
        compile_opts += ' --use_fast_math'
        nvcc_cmd = _prepare_nvcc_cli(compile_opts)

        # Hash build configuration.
        md5.update(f'nvcc_cmd: {nvcc_cmd}'.encode('utf-8') + b'\n')
        md5.update(f'tf.VERSION: {tf.VERSION}'.encode('utf-8') + b'\n')
        md5.update(
            f'cuda_cache_version_tag: {cuda_cache_version_tag}'.encode('utf-8')
            + b'\n'
        )

        # Compile if not already compiled.
        bin_file_ext = '.dll' if os.name == 'nt' else '.so'
        bin_file = os.path.join(
            cuda_cache_path,
            f'{cuda_file_name}_{md5.hexdigest()}{bin_file_ext}',
        )
        if not os.path.isfile(bin_file):
            if verbose:
                print('Compiling... ', end='', flush=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, f'{cuda_file_name}_tmp{bin_file_ext}')
                _run_cmd(
                    f'{nvcc_cmd} "{cuda_file}" --shared -o "{tmp_file}" --keep --keep-dir "{tmp_dir}"'
                )
                os.makedirs(cuda_cache_path, exist_ok=True)
                intermediate_file = os.path.join(
                    cuda_cache_path,
                    f'{cuda_file_name}_{uuid.uuid4().hex}_tmp{bin_file_ext}',
                )
                shutil.copyfile(tmp_file, intermediate_file)
                os.rename(intermediate_file, bin_file) # atomic

        # Load.
        if verbose:
            print('Loading... ', end='', flush=True)
        plugin = tf.load_op_library(bin_file)

        # Add to cache.
        _plugin_cache[cuda_file] = plugin
        if verbose:
            print('Done.', flush=True)
        return plugin

    except:
        if verbose:
            print('Failed!', flush=True)
        raise

#----------------------------------------------------------------------------
