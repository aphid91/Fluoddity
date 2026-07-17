"""OptiX path tracer: multi-bounce path tracing with three materials.

This module provides the PathTracerRenderer class, a self-contained renderer
that reads Fluoddity's entity SSBO via GL-CUDA interop and produces a
path-traced image with progressive sample accumulation and tonemapping.

Step 6: Motion blur + realtime/offline split. Adds render_realtime() for
single-frame renders with GAS scheduling, and render_offline_begin/substep/
finish for multi-substep motion-blur renders with temporal accumulation.
"""

import numpy as np

import optix
import cupy as cp
from cuda.bindings import runtime as cudart

import moderngl

from .interop import (
    check_cuda,
    register_gl_buffer,
    map_resource,
    unmap_resource,
    unregister_resource,
    compile_ptx,
    aligned_dtype,
    to_device,
)
from .cuda_src import PATHTRACER_CUDA_SRC, RESOLVE_TO_HALF_CUDA_SRC, RESOLVE_CUDA_SRC


# ---------------------------------------------------------------------------
# Launch-params dtype -- must match the CUDA Params struct field-by-field.
#
# CUDA struct layout (with alignment):
#   uchar4*            image;            // offset  0, ptr (8 bytes)
#   float*             entities;         // offset  8, ptr (8 bytes)
#   unsigned int       entity_stride;    // offset 16, u4  (4 bytes)
#   <pad 4 bytes>                        // offset 20
#   unsigned long long handle;           // offset 24, u8  (8 bytes)
#   unsigned int       width;            // offset 32, u4
#   unsigned int       height;           // offset 36, u4
#   float3 eye;                          // offset 40, 3×f4
#   float3 U;                            // offset 52, 3×f4
#   float3 V;                            // offset 64, 3×f4
#   float3 W;                            // offset 76, 3×f4
#   float3 sun_direction;                // offset 88, 3×f4 (Step 4: was light_dir)
#   float  sun_intensity;                // offset 100, f4  (Step 4: was ambient)
#   float  radius_scale;                 // offset 104, f4
#   float3 sky_color_top;                // offset 108, 3×f4
#   float3 sky_color_bottom;             // offset 120, 3×f4
#   <pad 4 bytes>                        // offset 132 (align ptr to 8)
#   float4* accum_buffer;                // offset 136, ptr (8 bytes)
#   unsigned int sample_index;           // offset 144, u4
#   unsigned int samples_accumulated;    // offset 148, u4
#   float  exposure;                     // offset 152, f4
#   --- Step 3 additions ---
#   float  aperture;                     // offset 156, f4
#   float  focal_plane_depth;            // offset 160, f4
#   float3 cam_right;                    // offset 164, 3×f4
#   float3 cam_up;                       // offset 176, 3×f4
#   int    max_bounces;                  // offset 188, i4
#   int    rr_start_depth;               // offset 192, i4
#   int    firefly_clamp;                // offset 196, i4
#   float  firefly_clamp_max;            // offset 200, f4
#   int    global_material;              // offset 204, i4
#   float  glossy_ior;                   // offset 208, f4
#   --- Step 4 additions ---
#   float3 sun_color;                    // offset 212, 3×f4
#   int    sun_sampling;                 // offset 224, i4
#   <pad 4 bytes>                        // offset 228 (align ptr to 8)
#   --- Step 5 additions ---
#   float4* albedo_buffer;               // offset 232, ptr (8 bytes)
#   float4* normal_buffer;               // offset 240, ptr (8 bytes)
#   Total: 248 bytes
# ---------------------------------------------------------------------------
PARAMS_DTYPE = np.dtype({
    "names": [
        "image", "entities", "entity_stride", "_pad0", "handle",
        "width", "height",
        "eye_x", "eye_y", "eye_z",
        "u_x", "u_y", "u_z",
        "v_x", "v_y", "v_z",
        "w_x", "w_y", "w_z",
        "sun_dir_x", "sun_dir_y", "sun_dir_z",
        "sun_intensity", "radius_scale",
        "sky_top_r", "sky_top_g", "sky_top_b",
        "sky_bot_r", "sky_bot_g", "sky_bot_b",
        "_pad1", "accum_buffer",
        "sample_index", "samples_accumulated",
        "exposure",
        # Step 3
        "aperture", "focal_plane_depth",
        "cam_right_x", "cam_right_y", "cam_right_z",
        "cam_up_x", "cam_up_y", "cam_up_z",
        "max_bounces", "rr_start_depth",
        "firefly_clamp", "firefly_clamp_max",
        "global_material", "glossy_ior",
        # Step 4
        "sun_color_r", "sun_color_g", "sun_color_b",
        "sun_sampling",
        "_pad4",
        # Step 5
        "albedo_buffer", "normal_buffer",
        # Albedo controls
        "albedo_saturation", "albedo_brightness",
        # Sphere size jitter
        "sphere_size_jitter",
        # RNG decorrelation
        "frame_seed",
        # SDF scene
        "sdf_enabled",
        "sdf_aabb_min_x", "sdf_aabb_min_y", "sdf_aabb_min_z",
        "sdf_aabb_max_x", "sdf_aabb_max_y", "sdf_aabb_max_z",
        "sdf_prim_index",
        # Emissive particles
        "emission_intensity",
        # Curve primitives
        "use_curves",
        "curve_length",
        "curve_r0",
        "curve_r1",
        "env_sky_nee",
        # Photosphere
        "photosphere_enabled",
        "_pad_photo",
        "photosphere_tex",
        "photosphere_avg_r",
        "photosphere_avg_g",
        "photosphere_avg_b",
        # Rasterize preset (merged sphere renderer)
        "rasterize",
        "ao_enabled", "ao_num_rays", "ao_radius", "ao_frame_index",
        "ambient_color_r", "ambient_color_g", "ambient_color_b",
        # Single-ray entity picking. pick_mode packs into the old trailing pad
        # at 380; pick_out (pointer) is 8-byte aligned at 384.
        "pick_mode", "pick_out",
        "pick_dir_x", "pick_dir_y", "pick_dir_z",
        # Cos-lobe sky sun sharpness (packs into old trailing pad at 404)
        "sun_exp",
    ],
    "formats": [
        "u8", "u8", "u4", "u4", "u8",
        "u4", "u4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "u4", "u8",
        "u4", "u4",
        "f4",
        # Step 3
        "f4", "f4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "i4", "i4",
        "i4", "f4",
        "i4", "f4",
        # Step 4
        "f4", "f4", "f4",
        "i4",
        "u4",
        # Step 5
        "u8", "u8",
        # Albedo controls
        "f4", "f4",
        # Sphere size jitter
        "f4",
        # RNG decorrelation
        "u4",
        # SDF scene
        "i4",
        "f4", "f4", "f4",
        "f4", "f4", "f4",
        "u4",
        # Emissive particles
        "f4",
        # Curve primitives
        "i4",
        "f4",
        "f4",
        "f4",
        "i4",
        # Photosphere
        "i4",
        "u4",
        "u8",
        "f4",
        "f4",
        "f4",
        # Rasterize preset
        "i4",
        "i4", "i4", "f4", "u4",
        "f4", "f4", "f4",
        # Single-ray entity picking
        "i4", "u8",
        "f4", "f4", "f4",
        # Cos-lobe sky sun sharpness
        "f4",
    ],
    "offsets": [
        0, 8, 16, 20, 24,
        32, 36,
        40, 44, 48,
        52, 56, 60,
        64, 68, 72,
        76, 80, 84,
        88, 92, 96,
        100, 104,
        108, 112, 116,
        120, 124, 128,
        132, 136,
        144, 148,
        152,
        # Step 3
        156, 160,
        164, 168, 172,
        176, 180, 184,
        188, 192,
        196, 200,
        204, 208,
        # Step 4
        212, 216, 220,
        224,
        228,
        # Step 5
        232, 240,
        # Albedo controls
        248, 252,
        # Sphere size jitter
        256,
        # RNG decorrelation
        260,
        # SDF scene
        264,
        268, 272, 276,
        280, 284, 288,
        292,
        # Emissive particles
        296,
        # Curve primitives
        300,
        304,
        308,
        312,
        316,
        # Photosphere
        320,
        324,
        328,
        336,
        340,
        344,
        # Rasterize preset
        348,
        352, 356, 360, 364,
        368, 372, 376,
        # Single-ray entity picking
        380, 384,
        392, 396, 400,
        # Cos-lobe sky sun sharpness
        404,
    ],
    "itemsize": 408,
})


def _camera_basis_from_vectors(pos, direction, up, fov_deg, aspect):
    """Convert FPS camera vectors to OptiX pinhole basis (eye, U, V, W).

    Args:
        pos: Camera position (3,).
        direction: Unit look direction (3,).
        up: Unit up vector (3,).
        fov_deg: Vertical field of view in degrees.
        aspect: Width / height.

    Returns:
        (eye, U, V, W) as numpy float32 arrays.
    """
    eye = np.asarray(pos, dtype=np.float32)
    W = np.asarray(direction, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    U = np.cross(W, up)
    U = U / max(np.linalg.norm(U), 1e-8)
    V = np.cross(U, W)
    V = V / max(np.linalg.norm(V), 1e-8)

    # Scale by FOV: at distance 1 along W, the half-height is tan(fov/2)
    vlen = np.tan(0.5 * np.radians(fov_deg))
    U = U * vlen * aspect
    V = V * vlen
    # W stays as unit direction (distance 1)

    return eye, U.astype(np.float32), V.astype(np.float32), W.astype(np.float32)


class PathTracerRenderer:
    """Path-trace entities as spheres using OptiX RT cores with HDR accumulation.

    Reads a ModernGL entity buffer (SSBO with 8-float stride) via GL-CUDA
    interop, builds a GAS over sphere AABBs, and renders into a GL texture
    via a PBO with progressive sample accumulation and tonemapping.

    Usage:
        renderer = PathTracerRenderer(ctx, entity_buffer, entity_count)
        renderer.build_accel()
        tex = renderer.render(width, height, eye, U, V, W, light_dir)
        # ... display tex ...
        renderer.cleanup()
    """

    def __init__(self, ctx, entity_buffer, entity_count, entity_stride=8):
        """Initialize OptiX context, compile PTX, build pipeline and SBT.

        Args:
            ctx: ModernGL context (must be current).
            entity_buffer: ModernGL buffer with Entity structs (SSBO binding 0).
            entity_count: Number of entities in the buffer.
            entity_stride: Floats per entity (default 8).
        """
        self._ctx = ctx
        self._entity_count = entity_count
        self._entity_stride = entity_stride

        # Display resources (lazily sized in render())
        self._pbo = None
        self._tex = None
        self._pbo_res = None
        self._render_width = 0
        self._render_height = 0

        # GAS resources (built in build_accel())
        self._gas_handle = None
        self._d_gas = None
        self._d_temp = None
        self._d_aabbs = None
        self._total_prims = 0
        self._temp_size = 0
        self._gas_size = 0

        # Entity picking (single-ray). Cached geometry params from the last
        # build_accel so pick() matches the sphere sizes / SDF prim in the GAS.
        self._d_pick_out = None
        self._last_radius_scale = 1.0
        self._last_sphere_size_jitter = 0.0
        self._sdf_enabled = False

        # Curve resources (built in _compute_curve_data())
        self._d_curve_vertices = None
        self._d_curve_widths = None
        self._d_curve_indices = None
        self._d_sdf_aabb = None

        # IAS resources (for curves + SDF: two GAS combined via IAS)
        self._sdf_gas_handle = None
        self._d_sdf_gas = None
        self._d_sdf_temp = None
        self._d_ias = None
        self._d_ias_temp = None
        self._d_instances = None
        self._ias_handle = None
        self._traversable_handle = None  # handle passed to optixTrace (GAS or IAS)

        # Accumulation resources
        self._d_accum = None
        self._d_albedo = None
        self._d_normal = None
        self._sample_count = 0
        self._accum_width = 0
        self._accum_height = 0

        # Accumulation slots: in stereogram accumulate mode each eye needs its
        # own persistent accumulation buffer (they occupy the same pixel
        # coordinates but see different cameras). A slot swaps only the accum
        # state in/out; the PBO/texture are transient scratch and shared.
        self._active_accum_slot = 0
        self._accum_slots = {}   # slot -> saved accum state dict

        # Denoiser resources (lazily initialized)
        self._denoiser = None
        self._denoiser_width = 0
        self._denoiser_height = 0
        self._d_denoiser_state = None
        self._d_denoiser_scratch = None
        self._denoiser_state_size = 0
        self._denoiser_scratch_size = 0
        self._d_resolved = None
        self._d_denoised = None

        # Monotonic frame counter for RNG decorrelation (never resets)
        self._global_frame_counter = 0

        # Photosphere CUDA texture
        self._photo_cuda_array = None         # cudaArray_t handle
        self._photo_tex_obj = 0               # cudaTextureObject_t (unsigned long long)
        self._photo_loaded = False

        # Offline render state
        self._offline_active = False
        self._offline_width = 0
        self._offline_height = 0
        self._offline_denoise = False
        self._offline_substeps_done = 0
        self._offline_spp_per_substep = 0

        # Initialize CuPy / CUDA primary context
        cp.zeros(1)

        # Register entity buffer with CUDA (read-only)
        self._entity_buffer = entity_buffer
        flags = cudart.cudaGraphicsRegisterFlags
        self._entity_res = register_gl_buffer(
            entity_buffer, flags.cudaGraphicsRegisterFlagsReadOnly
        )

        # OptiX context
        def _log_cb(level, tag, msg):
            print(f"[OptiX-PT {level:>2}][{tag:>12}]: {msg}")

        opts = optix.DeviceContextOptions(
            logCallbackFunction=_log_cb, logCallbackLevel=3
        )
        self._octx = optix.deviceContextCreate(0, opts)

        # Compile PTX via NVRTC
        self._ptx = self._load_ptx()
        self._use_curves = False  # current pipeline mode
        self._pipeline, self._groups = self._build_pipeline(self._ptx, use_curves=False)
        self._sbt, self._sbt_mem = self._build_sbt()

        # Device-side launch params buffer
        self._d_params = cp.empty(PARAMS_DTYPE.itemsize, dtype=cp.uint8)

        # Dedicated CUDA stream for OptiX launches
        self._stream_obj = check_cuda(cudart.cudaStreamCreate())
        self._stream = int(self._stream_obj)  # raw handle for OptiX/CUDA APIs

        # CUDA events for timing instrumentation
        self._evt_gas_start = check_cuda(cudart.cudaEventCreate())
        self._evt_gas_end = check_cuda(cudart.cudaEventCreate())
        self._evt_render_start = check_cuda(cudart.cudaEventCreate())
        self._evt_render_end = check_cuda(cudart.cudaEventCreate())
        self.last_gas_ms = 0.0
        self.last_render_ms = 0.0

        # Compile resolve kernels via CuPy RawKernel
        self._resolve_to_half_kernel = cp.RawKernel(
            RESOLVE_TO_HALF_CUDA_SRC, 'resolve_to_half')
        self._resolve_kernel = cp.RawKernel(RESOLVE_CUDA_SRC, 'resolve')

        # Log device info
        err, props = cudart.cudaGetDeviceProperties(0)
        if err == cudart.cudaError_t.cudaSuccess:
            name = props.name
            if isinstance(name, bytes):
                name = name.decode().rstrip('\x00')
            print(f"PathTracer: CUDA device: {name}")

    # ------------------------------------------------------------------
    # PTX loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ptx():
        """Load pre-compiled PTX if available, otherwise compile via NVRTC."""
        import os
        ptx_path = os.path.join(
            os.path.dirname(__file__), "pathtracer.ptx"
        )
        if os.path.isfile(ptx_path):
            with open(ptx_path, "rb") as f:
                ptx = f.read()
            print(f"PathTracer: loaded pre-compiled PTX ({len(ptx)} bytes)")
            return ptx

        print("PathTracer: compiling PTX via NVRTC (no pre-compiled PTX found)...")
        return compile_ptx(PATHTRACER_CUDA_SRC)

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def reload_shaders(self):
        """Recompile CUDA source and rebuild OptiX pipeline (hot reload).

        Re-reads sdf_scene.py and cuda_src.py from disk via importlib.reload(),
        recompiles PTX via NVRTC, and rebuilds the OptiX module/pipeline/SBT.
        """
        import importlib
        from . import sdf_scene as _sdf_mod
        from . import cuda_src as _cuda_mod

        # 1. Reload Python modules to pick up disk edits
        #    Order matters: cuda_src imports from sdf_scene at module level
        importlib.reload(_sdf_mod)
        importlib.reload(_cuda_mod)

        # 2. Re-read the assembled source and recompile PTX
        from .cuda_src import PATHTRACER_CUDA_SRC as fresh_src
        print("PathTracer: recompiling PTX via NVRTC...")
        self._ptx = compile_ptx(fresh_src)
        print(f"PathTracer: PTX compiled ({len(self._ptx)} bytes)")

        # 3. Rebuild pipeline + SBT with current curve mode
        self._pipeline, self._groups = self._build_pipeline(
            self._ptx, use_curves=self._use_curves)
        self._sbt, self._sbt_mem = self._build_sbt()

        # 4. Invalidate GAS so it rebuilds next frame
        self._gas_handle = None
        self._traversable_handle = None
        self._sdf_gas_handle = None
        self._ias_handle = None

        print("PathTracer: hot reload complete")

    # ------------------------------------------------------------------
    # Pipeline / SBT construction
    # ------------------------------------------------------------------

    def _build_pipeline(self, ptx, use_curves=False):
        """Create OptiX pipeline with raygen, miss, and hitgroup programs.

        When use_curves is True, includes both custom (SDF) and curve (entity)
        primitive types, with a built-in IS module for round linear curves.
        """
        prim_flags = optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
        if use_curves:
            prim_flags = int(prim_flags) | int(optix.PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR)

        # When curves are enabled, we may need an IAS (curves GAS + SDF GAS)
        # so use ALLOW_ANY to support both bare GAS and IAS traversal.
        if use_curves:
            trav_flags = int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY)
        else:
            trav_flags = int(optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS)

        pco = optix.PipelineCompileOptions(
            usesMotionBlur=False,
            traversableGraphFlags=trav_flags,
            numPayloadValues=8,
            numAttributeValues=2,
            exceptionFlags=int(optix.EXCEPTION_FLAG_NONE),
            pipelineLaunchParamsVariableName="params",
            usesPrimitiveTypeFlags=prim_flags,
        )

        mco = optix.ModuleCompileOptions(
            maxRegisterCount=optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel=optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel=optix.COMPILE_DEBUG_LEVEL_DEFAULT,
        )
        module, _log = self._octx.moduleCreate(mco, pco, ptx)

        # Raygen
        rg_desc = optix.ProgramGroupDesc()
        rg_desc.raygenModule = module
        rg_desc.raygenEntryFunctionName = "__raygen__rg"
        (rg,), _ = self._octx.programGroupCreate([rg_desc])

        # Miss: radiance (index 0)
        ms_desc = optix.ProgramGroupDesc()
        ms_desc.missModule = module
        ms_desc.missEntryFunctionName = "__miss__radiance"
        (ms_rad,), _ = self._octx.programGroupCreate([ms_desc])

        # Miss: occlusion (index 1)
        ms2_desc = optix.ProgramGroupDesc()
        ms2_desc.missModule = module
        ms2_desc.missEntryFunctionName = "__miss__occlusion"
        (ms_occ,), _ = self._octx.programGroupCreate([ms2_desc])

        # Hit group: closest-hit + custom intersection (spheres + SDF)
        hg_desc = optix.ProgramGroupDesc()
        hg_desc.hitgroupModuleCH = module
        hg_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
        hg_desc.hitgroupModuleIS = module
        hg_desc.hitgroupEntryFunctionNameIS = "__intersection__sphere"
        (hg,), _ = self._octx.programGroupCreate([hg_desc])

        groups = [rg, ms_rad, ms_occ, hg]

        # Hit group for curves (built-in IS + custom closest-hit)
        if use_curves:
            bis_opts = optix.BuiltinISOptions()
            bis_opts.builtinISModuleType = optix.PrimitiveType.PRIMITIVE_TYPE_ROUND_LINEAR
            bis_opts.buildFlags = int(
                optix.BUILD_FLAG_PREFER_FAST_TRACE
                | optix.BUILD_FLAG_ALLOW_UPDATE
                | optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
            )
            curve_module = self._octx.builtinISModuleGet(mco, pco, bis_opts)

            hg_curve_desc = optix.ProgramGroupDesc()
            hg_curve_desc.hitgroupModuleCH = module
            hg_curve_desc.hitgroupEntryFunctionNameCH = "__closesthit__curve"
            hg_curve_desc.hitgroupModuleIS = curve_module
            # Built-in IS module: do NOT set hitgroupEntryFunctionNameIS
            # (default null tells OptiX to use the built-in intersection)
            (hg_curve,), _ = self._octx.programGroupCreate([hg_curve_desc])
            groups.append(hg_curve)

        max_trace_depth = 2  # primary + shadow (shadow added in Step 4)
        plo = optix.PipelineLinkOptions()
        plo.maxTraceDepth = max_trace_depth
        pipeline = self._octx.pipelineCreate(pco, plo, groups, "")

        # Configure stack sizes
        stack = optix.StackSizes()
        for g in groups:
            optix.util.accumulateStackSizes(g, stack, pipeline)
        dc_trav, dc_state, cc = optix.util.computeStackSizes(
            stack, max_trace_depth, 0, 0
        )
        pipeline.setStackSize(dc_trav, dc_state, cc, 1)

        return pipeline, groups

    def _build_sbt(self):
        """Build the Shader Binding Table.

        When curves are enabled (groups has 5 entries), the SBT has two
        hit group records: [0] = curves (for BuildInputCurveArray geometry),
        [1] = custom IS (for spheres/SDF, BuildInputCustomPrimitiveArray).
        When curves are disabled (groups has 4 entries), single HG record.
        """
        has_curves = len(self._groups) == 5
        if has_curves:
            rg, ms_rad, ms_occ, hg, hg_curve = self._groups
        else:
            rg, ms_rad, ms_occ, hg = self._groups

        hdr = f"{optix.SBT_RECORD_HEADER_SIZE}B"
        dtype = aligned_dtype(["header"], [hdr], optix.SBT_RECORD_ALIGNMENT)

        h_rg = np.zeros(1, dtype=dtype)
        optix.sbtRecordPackHeader(rg, h_rg)

        h_ms0 = np.zeros(1, dtype=dtype)
        h_ms1 = np.zeros(1, dtype=dtype)
        optix.sbtRecordPackHeader(ms_rad, h_ms0)
        optix.sbtRecordPackHeader(ms_occ, h_ms1)
        h_ms = np.concatenate([h_ms0, h_ms1])

        if has_curves:
            # Two HG records: [0] = curves, [1] = custom (SDF)
            h_hg0 = np.zeros(1, dtype=dtype)
            h_hg1 = np.zeros(1, dtype=dtype)
            optix.sbtRecordPackHeader(hg_curve, h_hg0)
            optix.sbtRecordPackHeader(hg, h_hg1)
            h_hg = np.concatenate([h_hg0, h_hg1])
            hg_count = 2
        else:
            h_hg = np.zeros(1, dtype=dtype)
            optix.sbtRecordPackHeader(hg, h_hg)
            hg_count = 1

        d_rg = to_device(h_rg)
        d_ms = to_device(h_ms)
        d_hg = to_device(h_hg)

        sbt = optix.ShaderBindingTable(
            raygenRecord=d_rg.ptr,
            missRecordBase=d_ms.ptr,
            missRecordStrideInBytes=dtype.itemsize,
            missRecordCount=2,
            hitgroupRecordBase=d_hg.ptr,
            hitgroupRecordStrideInBytes=dtype.itemsize,
            hitgroupRecordCount=hg_count,
        )
        return sbt, (d_rg, d_ms, d_hg)  # prevent GC

    def _ensure_pipeline(self, use_curves):
        """Rebuild OptiX pipeline and SBT if curve mode changed."""
        if use_curves == self._use_curves:
            return
        self._use_curves = use_curves
        self._pipeline, self._groups = self._build_pipeline(
            self._ptx, use_curves=use_curves)
        self._sbt, self._sbt_mem = self._build_sbt()
        # Force full GAS rebuild since geometry type changed
        self._gas_handle = None
        self._traversable_handle = None
        self._sdf_gas_handle = None
        self._ias_handle = None

    # ------------------------------------------------------------------
    # Display resources (PBO + texture)
    # ------------------------------------------------------------------

    def _ensure_display(self, width, height):
        """Create or resize the PBO and output texture."""
        if self._render_width == width and self._render_height == height:
            return

        # Clean up old PBO registration
        if self._pbo_res is not None:
            unregister_resource(self._pbo_res)
            self._pbo_res = None

        # Create new PBO and texture (rgba16f for linear HDR output)
        self._pbo = self._ctx.buffer(
            reserve=width * height * 8, dynamic=True  # 4 x fp16 per pixel
        )
        self._tex = self._ctx.texture((width, height), 4, dtype='f2')
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Register PBO with CUDA (write-discard)
        flags = cudart.cudaGraphicsRegisterFlags
        self._pbo_res = register_gl_buffer(
            self._pbo, flags.cudaGraphicsRegisterFlagsWriteDiscard
        )

        self._render_width = width
        self._render_height = height

    # ------------------------------------------------------------------
    # Accumulation buffer
    # ------------------------------------------------------------------

    def _ensure_accum_buffers(self, width, height):
        """Allocate or resize CUDA-side float4 accumulation and guide buffers."""
        if self._accum_width == width and self._accum_height == height:
            return

        self._d_accum = cp.zeros(width * height * 4, dtype=cp.float32)
        self._d_albedo = cp.zeros(width * height * 4, dtype=cp.float32)
        self._d_normal = cp.zeros(width * height * 4, dtype=cp.float32)
        self._accum_width = width
        self._accum_height = height
        self._sample_count = 0

    def reset_accumulation(self):
        """Zero the accumulation and guide buffers, reset the sample counter."""
        if self._d_accum is not None:
            self._d_accum.fill(0)
        if self._d_albedo is not None:
            self._d_albedo.fill(0)
        if self._d_normal is not None:
            self._d_normal.fill(0)
        self._sample_count = 0

    # -- Accumulation slots (stereogram: one persistent accumulator per eye) --

    _ACCUM_STATE_FIELDS = (
        '_d_accum', '_d_albedo', '_d_normal',
        '_sample_count', '_accum_width', '_accum_height',
        # Offline substep progress is per-eye too (stereo interleaves eyes).
        '_offline_substeps_done',
    )

    def _snapshot_accum_state(self):
        return {f: getattr(self, f) for f in self._ACCUM_STATE_FIELDS}

    def _restore_accum_state(self, state):
        for f in self._ACCUM_STATE_FIELDS:
            setattr(self, f, state[f])

    def reset_all_accum_slots(self):
        """Reset the active accumulator and drop all other slots' saved state.

        In stereogram accumulate mode each eye owns a slot; a camera move must
        clear every eye, not just the one that happened to render last. Dropped
        slots reallocate zeroed on next use. For non-stereo (only slot 0) this
        is equivalent to ``reset_accumulation``.
        """
        self.reset_accumulation()
        self._accum_slots.clear()

    def set_accum_slot(self, slot):
        """Activate accumulation buffer `slot`, preserving each slot's state.

        Used by stereogram accumulate mode so each eye accumulates into its own
        persistent buffer instead of blending into a shared one. Slot 0 is the
        default (non-stereo) accumulator. Switching slots stashes the current
        accum state and swaps in the target slot's; `_ensure_accum_buffers`
        lazily allocates a fresh (zeroed) buffer the first time a slot is used.
        """
        if slot == self._active_accum_slot:
            return
        # Stash current slot's live state.
        self._accum_slots[self._active_accum_slot] = self._snapshot_accum_state()
        # Restore target slot's state, or start empty (forces reallocation).
        saved = self._accum_slots.get(slot)
        if saved is not None:
            self._restore_accum_state(saved)
        else:
            # Fresh slot: null the buffers (forces _ensure_accum_buffers to
            # reallocate zeroed) and reset all per-slot counters.
            self._d_accum = None
            self._d_albedo = None
            self._d_normal = None
            self._sample_count = 0
            self._accum_width = 0
            self._accum_height = 0
            self._offline_substeps_done = 0
        self._active_accum_slot = slot

    # ------------------------------------------------------------------
    # Denoiser (Step 5)
    # ------------------------------------------------------------------

    def _setup_denoiser(self, width, height):
        """Create and setup the OptiX AI denoiser for the given resolution.

        Lazily called on first denoise request. Re-created if resolution changes.
        """
        if (self._denoiser is not None
                and self._denoiser_width == width
                and self._denoiser_height == height):
            return

        # Destroy previous denoiser if resolution changed
        self._denoiser = None
        self._d_denoiser_state = None
        self._d_denoiser_scratch = None

        # Create denoiser with albedo + normal guides
        opts = optix.DenoiserOptions()
        opts.guideAlbedo = 1
        opts.guideNormal = 1
        self._denoiser = self._octx.denoiserCreate(
            optix.DENOISER_MODEL_KIND_HDR, opts
        )

        # Compute memory requirements
        sizes = self._denoiser.computeMemoryResources(width, height)
        self._denoiser_state_size = sizes.stateSizeInBytes
        self._denoiser_scratch_size = sizes.withoutOverlapScratchSizeInBytes

        # Allocate state and scratch buffers
        self._d_denoiser_state = cp.cuda.alloc(self._denoiser_state_size)
        self._d_denoiser_scratch = cp.cuda.alloc(self._denoiser_scratch_size)

        # Setup denoiser
        self._denoiser.setup(
            self._stream,
            width, height,
            self._d_denoiser_state.ptr, self._denoiser_state_size,
            self._d_denoiser_scratch.ptr, self._denoiser_scratch_size,
        )

        # Allocate resolved (accum / N) and denoised output buffers
        self._d_resolved = cp.zeros(width * height * 4, dtype=cp.float32)
        self._d_denoised = cp.zeros(width * height * 4, dtype=cp.float32)

        self._denoiser_width = width
        self._denoiser_height = height
        print(f"PathTracer: denoiser initialized ({width}x{height}, "
              f"state={self._denoiser_state_size} bytes, "
              f"scratch={self._denoiser_scratch_size} bytes)")

    @staticmethod
    def _make_image2d(ptr, width, height):
        """Build an OptiX Image2D descriptor for a float4 buffer."""
        img = optix.Image2D()
        img.data = ptr
        img.width = width
        img.height = height
        img.format = optix.PIXEL_FORMAT_FLOAT4
        img.pixelStrideInBytes = 16      # float4 = 4 * 4 bytes
        img.rowStrideInBytes = width * 16
        return img

    def _run_denoiser(self, width, height):
        """Resolve accumulation buffer and run the OptiX denoiser.

        Produces a denoised HDR image in self._d_denoised.
        """
        # 1. Resolve: divide accum by sample count
        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16, 1)
        stream_wrapper = cp.cuda.ExternalStream(self._stream)
        self._resolve_kernel(
            grid, block,
            (self._d_accum.data.ptr,
             self._d_resolved.data.ptr,
             np.uint32(width),
             np.uint32(height),
             np.uint32(self._sample_count)),
            stream=stream_wrapper,
        )

        # 2. Build Image2D descriptors
        input_img = self._make_image2d(
            self._d_resolved.data.ptr, width, height)
        output_img = self._make_image2d(
            self._d_denoised.data.ptr, width, height)
        albedo_img = self._make_image2d(
            self._d_albedo.data.ptr, width, height)
        normal_img = self._make_image2d(
            self._d_normal.data.ptr, width, height)

        # 3. Build guide and layer structs
        guide = optix.DenoiserGuideLayer()
        guide.albedo = albedo_img
        guide.normal = normal_img

        layer = optix.DenoiserLayer()
        layer.input = input_img
        layer.output = output_img

        # 4. Denoiser params
        dn_params = optix.DenoiserParams()
        dn_params.blendFactor = 0.0   # full denoise
        dn_params.hdrIntensity = 0    # skip intensity computation

        # 5. Invoke denoiser
        # C API order: stream, params, state, stateSize, guide, layer, numLayers,
        #              offsetX, offsetY, scratch, scratchSize
        self._denoiser.invoke(
            self._stream,
            dn_params,
            self._d_denoiser_state.ptr, self._denoiser_state_size,
            guide, layer, 1,
            0, 0,
            self._d_denoiser_scratch.ptr, self._denoiser_scratch_size,
        )

    # ------------------------------------------------------------------
    # Acceleration structure (GAS)
    # ------------------------------------------------------------------

    def _compute_aabbs(self, d_entities_ptr, radius_scale=1.0,
                       sphere_size_jitter=0.0,
                       sdf_enabled=False, sdf_aabb_min=None, sdf_aabb_max=None):
        """Compute float6 AABBs from the mapped entity buffer via CuPy.

        When sdf_enabled, appends one extra AABB for the SDF scene.
        """
        nbytes = self._entity_count * self._entity_stride * 4
        mem = cp.cuda.UnownedMemory(d_entities_ptr, nbytes, owner=None)
        entities_flat = cp.ndarray(
            self._entity_count * self._entity_stride,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(mem, 0),
        )
        entities = entities_flat.reshape(self._entity_count, self._entity_stride)

        pos = entities[:, 0:3]
        rad = entities[:, 7:8] * radius_scale

        # Apply per-sphere jitter matching the CUDA intersection shader
        if sphere_size_jitter > 0.0:
            h = (cp.arange(self._entity_count, dtype=cp.uint32) * 2654435761) & 0xFFFF
            jitter = h.astype(cp.float32) / 32767.5 - 1.0
            rad = rad * (1.0 + sphere_size_jitter * jitter.reshape(-1, 1))

        total_prims = self._entity_count + (1 if sdf_enabled else 0)
        if self._d_aabbs is None or self._d_aabbs.shape[0] != total_prims:
            self._d_aabbs = cp.empty(
                (total_prims, 6), dtype=cp.float32
            )

        self._d_aabbs[:self._entity_count, 0:3] = pos - rad
        self._d_aabbs[:self._entity_count, 3:6] = pos + rad

        # Invalidate AABBs for zero-size entities (min > max tells OptiX to cull)
        dead = (rad <= 0.0).ravel()
        if dead.any():
            self._d_aabbs[:self._entity_count][dead, 0:3] = 1.0
            self._d_aabbs[:self._entity_count][dead, 3:6] = -1.0

        # Append SDF AABB as the last primitive
        if sdf_enabled and sdf_aabb_min is not None:
            sdf_row = cp.array([[
                sdf_aabb_min[0], sdf_aabb_min[1], sdf_aabb_min[2],
                sdf_aabb_max[0], sdf_aabb_max[1], sdf_aabb_max[2],
            ]], dtype=cp.float32)
            self._d_aabbs[self._entity_count:self._entity_count + 1, :] = sdf_row

        self._total_prims = total_prims
        cp.cuda.Device().synchronize()

    def _compute_curve_data(self, d_entities_ptr, radius_scale=1.0,
                            curve_length=1.0, curve_r0=1.0, curve_r1=0.5):
        """Compute curve control points and widths from the entity buffer.

        For ROUND_LINEAR curves: 2 control points per entity placed along
        velocity direction, with per-vertex widths.

        The vertex buffer is (entity_count * 2, 3) float32 — pairs of float3.
        The width buffer is (entity_count * 2,) float32 — pairs of radius.
        The index buffer is (entity_count,) uint32 — start vertex per curve.
        """
        nbytes = self._entity_count * self._entity_stride * 4
        mem = cp.cuda.UnownedMemory(d_entities_ptr, nbytes, owner=None)
        entities_flat = cp.ndarray(
            self._entity_count * self._entity_stride,
            dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(mem, 0),
        )
        entities = entities_flat.reshape(self._entity_count, self._entity_stride)

        pos = entities[:, 0:3]   # (N, 3) position
        vel = entities[:, 3:6]   # (N, 3) velocity
        size = entities[:, 7:8] * radius_scale  # (N, 1) base radius

        # Normalize velocity with zero-safe fallback
        vel_len = cp.linalg.norm(vel, axis=1, keepdims=True)  # (N, 1)
        safe_len = cp.maximum(vel_len, 1e-8)
        vel_norm = vel / safe_len  # (N, 3)
        has_vel = (vel_len > 1e-6).astype(cp.float32)  # (N, 1)

        # Control points along velocity
        half_extent = has_vel * curve_length * size * 0.5  # (N, 1)
        cp0 = pos - vel_norm * half_extent  # (N, 3) start
        cp1 = pos + vel_norm * half_extent  # (N, 3) end

        # Widths (OptiX curves use diameter, not radius — multiply by 2)
        w0 = (size * curve_r0 * 2.0).ravel()  # (N,)
        w1 = (size * curve_r1 * 2.0).ravel()  # (N,)

        # Zero-size entities: collapse control points and set widths to 0
        # so OptiX builds a zero-volume degenerate curve (never intersected)
        dead = (size <= 0.0).ravel()
        if dead.any():
            cp0[dead] = 0.0
            cp1[dead] = 0.0
            w0[dead] = 0.0
            w1[dead] = 0.0

        # Interleave into vertex buffer: [cp0_0, cp1_0, cp0_1, cp1_1, ...]
        n = self._entity_count
        if self._d_curve_vertices is None or self._d_curve_vertices.shape[0] != n * 2:
            self._d_curve_vertices = cp.empty((n * 2, 3), dtype=cp.float32)
            self._d_curve_widths = cp.empty(n * 2, dtype=cp.float32)
            self._d_curve_indices = cp.arange(0, n * 2, 2, dtype=cp.uint32)

        self._d_curve_vertices[0::2, :] = cp0
        self._d_curve_vertices[1::2, :] = cp1
        self._d_curve_widths[0::2] = w0
        self._d_curve_widths[1::2] = w1

        cp.cuda.Device().synchronize()

    def _build_curve_input(self):
        """Construct a BuildInputCurveArray for entity curves."""
        return optix.BuildInputCurveArray(
            curveType=optix.PrimitiveType.PRIMITIVE_TYPE_ROUND_LINEAR,
            vertexBuffers=[self._d_curve_vertices.data.ptr],
            vertexStrideInBytes=12,
            numVertices=self._entity_count * 2,
            widthBuffers=[self._d_curve_widths.data.ptr],
            widthStrideInBytes=4,
            indexBuffer=self._d_curve_indices.data.ptr,
            indexStrideInBytes=4,
            numPrimitives=self._entity_count,
            flag=optix.GEOMETRY_FLAG_DISABLE_ANYHIT,
        )

    def _build_sdf_input(self):
        """Construct a BuildInputCustomPrimitiveArray for the SDF AABB."""
        return optix.BuildInputCustomPrimitiveArray(
            aabbBuffers=[self._d_sdf_aabb.data.ptr],
            numPrimitives=1,
            flags=[optix.GEOMETRY_FLAG_DISABLE_ANYHIT],
            numSbtRecords=1,
        )

    def _build_sphere_input(self):
        """Construct a BuildInputCustomPrimitiveArray for all sphere/SDF AABBs."""
        return optix.BuildInputCustomPrimitiveArray(
            aabbBuffers=[self._d_aabbs.data.ptr],
            numPrimitives=self._total_prims,
            flags=[optix.GEOMETRY_FLAG_DISABLE_ANYHIT],
            numSbtRecords=1,
        )

    def _accel_build_single(self, build_input, operation, use_curves=False):
        """Build or update a single GAS and return (handle, d_gas, d_temp, sizes).

        For BUILD operations, allocates new buffers.
        For UPDATE operations, reuses existing buffers from the returned tuple.
        """
        build_flags = int(
            optix.BUILD_FLAG_PREFER_FAST_TRACE
            | optix.BUILD_FLAG_ALLOW_UPDATE
        )
        if use_curves:
            build_flags |= int(optix.BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS)
        accel_opts = optix.AccelBuildOptions(
            buildFlags=build_flags,
            operation=operation,
        )

        if operation == optix.BUILD_OPERATION_BUILD:
            sizes = self._octx.accelComputeMemoryUsage(
                [accel_opts], [build_input]
            )
            d_temp = cp.cuda.alloc(sizes.tempSizeInBytes)
            d_gas = cp.cuda.alloc(sizes.outputSizeInBytes)
            temp_size = sizes.tempSizeInBytes
            gas_size = sizes.outputSizeInBytes
        else:
            # UPDATE reuses existing buffers — caller must pass them
            raise ValueError("Use _accel_update_single for UPDATE")

        handle = self._octx.accelBuild(
            self._stream, [accel_opts], [build_input],
            d_temp.ptr, temp_size, d_gas.ptr, gas_size, [],
        )
        return handle, d_gas, d_temp, temp_size, gas_size

    def _build_ias(self, gas_handles, sbt_offsets):
        """Build an Instance Acceleration Structure from multiple GAS handles.

        Args:
            gas_handles: List of GAS traversable handles.
            sbt_offsets: List of SBT hit group record offsets per instance.

        Returns:
            IAS traversable handle.
        """
        identity = [1.0, 0.0, 0.0, 0.0,
                     0.0, 1.0, 0.0, 0.0,
                     0.0, 0.0, 1.0, 0.0]
        instances = []
        for i, (gh, sbt_off) in enumerate(zip(gas_handles, sbt_offsets)):
            inst = optix.Instance(
                transform=identity,
                instanceId=i,
                sbtOffset=sbt_off,
                visibilityMask=255,
                flags=int(optix.INSTANCE_FLAG_NONE),
                traversableHandle=gh,
            )
            instances.append(inst)

        # Serialize to bytes and upload to device
        instance_bytes = optix.getDeviceRepresentation(instances)
        self._d_instances = cp.array(
            np.frombuffer(instance_bytes, dtype=np.uint8))

        ias_input = optix.BuildInputInstanceArray()
        ias_input.instances = self._d_instances.data.ptr
        ias_input.numInstances = len(instances)

        accel_opts = optix.AccelBuildOptions(
            buildFlags=int(optix.BUILD_FLAG_PREFER_FAST_TRACE),
            operation=optix.BUILD_OPERATION_BUILD,
        )

        sizes = self._octx.accelComputeMemoryUsage(
            [accel_opts], [ias_input]
        )
        self._d_ias_temp = cp.cuda.alloc(sizes.tempSizeInBytes)
        self._d_ias = cp.cuda.alloc(sizes.outputSizeInBytes)
        self._ias_size = sizes.outputSizeInBytes

        self._ias_handle = self._octx.accelBuild(
            self._stream, [accel_opts], [ias_input],
            self._d_ias_temp.ptr, sizes.tempSizeInBytes,
            self._d_ias.ptr, self._ias_size, [],
        )
        return self._ias_handle

    def build_accel(self, radius_scale=1.0, sphere_size_jitter=0.0,
                    sdf_enabled=False, sdf_aabb_min=None, sdf_aabb_max=None,
                    use_curves=False, curve_length=1.0, curve_r0=1.0,
                    curve_r1=0.5):
        """Full GAS build. Maps entity buffer, computes geometry, builds BVH.

        Must be called at least once before render(). Call again periodically
        to maintain BVH quality as entities move.

        When curves + SDF are both active, builds two separate GAS (one for
        curves, one for the SDF custom primitive) and combines them via an IAS,
        since OptiX requires all build inputs in a GAS to be the same type.

        Args:
            radius_scale: Multiplier on entity size for AABB computation.
            sphere_size_jitter: Per-sphere radius jitter magnitude (0-1).
            sdf_enabled: If True, include SDF geometry.
            sdf_aabb_min: SDF bounding box min (3-tuple).
            sdf_aabb_max: SDF bounding box max (3-tuple).
            use_curves: If True, build entity geometry as round linear curves.
            curve_length: Curve control point distance multiplier.
            curve_r0: Curve start radius multiplier.
            curve_r1: Curve end radius multiplier.
        """
        # Cache geometry params so pick() can match the sphere sizes / SDF prim
        # that this GAS was built with.
        self._last_radius_scale = radius_scale
        self._last_sphere_size_jitter = sphere_size_jitter
        self._sdf_enabled = sdf_enabled

        self._ensure_pipeline(use_curves)
        self._ctx.finish()
        entities_ptr, _ = map_resource(self._entity_res)

        try:
            check_cuda(cudart.cudaEventRecord(self._evt_gas_start, self._stream_obj))

            if use_curves:
                self._compute_curve_data(entities_ptr, radius_scale,
                                         curve_length, curve_r0, curve_r1)

                # Build curve GAS
                curve_input = self._build_curve_input()
                curve_handle, self._d_gas, self._d_temp, \
                    self._temp_size, self._gas_size = \
                    self._accel_build_single(curve_input, optix.BUILD_OPERATION_BUILD,
                                             use_curves=True)
                self._gas_handle = curve_handle

                if sdf_enabled and sdf_aabb_min is not None:
                    # Build separate SDF GAS
                    self._d_sdf_aabb = cp.array([[
                        sdf_aabb_min[0], sdf_aabb_min[1], sdf_aabb_min[2],
                        sdf_aabb_max[0], sdf_aabb_max[1], sdf_aabb_max[2],
                    ]], dtype=cp.float32)
                    sdf_input = self._build_sdf_input()
                    self._sdf_gas_handle, self._d_sdf_gas, self._d_sdf_temp, \
                        _, _ = \
                        self._accel_build_single(sdf_input, optix.BUILD_OPERATION_BUILD)

                    # Combine into IAS: instance 0 = curves (SBT offset 0),
                    #                   instance 1 = SDF (SBT offset 1)
                    self._traversable_handle = self._build_ias(
                        [curve_handle, self._sdf_gas_handle],
                        [0, 1],
                    )
                else:
                    # Curves only — bare GAS traversable
                    self._traversable_handle = curve_handle
                    self._sdf_gas_handle = None
            else:
                # Sphere mode: single GAS with all custom primitives
                self._compute_aabbs(entities_ptr, radius_scale,
                                    sphere_size_jitter, sdf_enabled,
                                    sdf_aabb_min, sdf_aabb_max)
                sphere_input = self._build_sphere_input()
                self._gas_handle, self._d_gas, self._d_temp, \
                    self._temp_size, self._gas_size = \
                    self._accel_build_single(sphere_input, optix.BUILD_OPERATION_BUILD)
                self._traversable_handle = self._gas_handle
                self._sdf_gas_handle = None

            check_cuda(cudart.cudaEventRecord(self._evt_gas_end, self._stream_obj))
            check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))
            self.last_gas_ms = check_cuda(
                cudart.cudaEventElapsedTime(self._evt_gas_start, self._evt_gas_end)
            )
        finally:
            unmap_resource(self._entity_res)

    # ------------------------------------------------------------------
    # Rendering helpers (Step 6: extracted for realtime/offline reuse)
    # ------------------------------------------------------------------

    def _fill_params_and_launch(self, entities_ptr, width, height,
                                eye, U, V, W, write_guides=True,
                                sun_direction=(0.577, 0.577, 0.577),
                                sun_intensity=1.0, radius_scale=1.0,
                                exposure=1.0,
                                sky_color_top=(0.45, 0.62, 0.85),
                                sky_color_bottom=(0.08, 0.08, 0.10),
                                aperture=0.0, focal_plane_depth=10.0,
                                cam_right=None, cam_up=None,
                                max_bounces=0, rr_start_depth=3,
                                firefly_clamp=False, firefly_clamp_max=100.0,
                                global_material=0, glossy_ior=1.5,
                                sun_color=(1.0, 1.0, 1.0), sun_sampling=True,
                                albedo_saturation=0.8, albedo_brightness=1.0,
                                sphere_size_jitter=0.0,
                                sdf_enabled=False, sdf_aabb_min=None,
                                sdf_aabb_max=None,
                                emission_intensity=10.0,
                                use_curves=False, curve_length=1.0,
                                curve_r0=1.0, curve_r1=0.5,
                                env_sky_nee=False,
                                sun_exp=15.0,
                                photosphere=False,
                                photosphere_avg_r=0.0,
                                photosphere_avg_g=0.0,
                                photosphere_avg_b=0.0,
                                rasterize=False,
                                ao_enabled=False, ao_num_rays=2,
                                ao_radius=0.5, ao_frame_index=0,
                                ambient_color=(0.12, 0.12, 0.12)):
        """Fill launch params and trace one sample (1 SPP) into the HDR buffer.

        The entity buffer must already be mapped (entities_ptr is the device
        pointer). Accumulation buffers must already be allocated. Does NOT
        map/unmap anything. Does NOT tonemap or denoise.

        Increments self._sample_count after the launch.

        Args:
            entities_ptr: Device pointer to the mapped entity buffer.
            width, height: Render dimensions.
            eye, U, V, W: Camera basis vectors.
            write_guides: If True, set albedo/normal buffer pointers so the
                raygen writes denoiser guide data. If False, pass null
                pointers so guide writes are skipped.
            ... (all other render params forwarded to the Params struct)
        """
        eye = np.asarray(eye, dtype=np.float32)
        U = np.asarray(U, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        W = np.asarray(W, dtype=np.float32)
        sun_direction = np.asarray(sun_direction, dtype=np.float32)
        sun_color = np.asarray(sun_color, dtype=np.float32)
        sky_color_top = np.asarray(sky_color_top, dtype=np.float32)
        sky_color_bottom = np.asarray(sky_color_bottom, dtype=np.float32)

        h_params = np.zeros(1, dtype=PARAMS_DTYPE)
        h_params["image"] = 0  # unused by raygen; tonemap uses its own arg
        h_params["entities"] = entities_ptr
        h_params["entity_stride"] = self._entity_stride
        h_params["_pad0"] = 0
        h_params["handle"] = self._traversable_handle
        h_params["width"] = width
        h_params["height"] = height
        h_params["eye_x"] = eye[0]
        h_params["eye_y"] = eye[1]
        h_params["eye_z"] = eye[2]
        h_params["u_x"] = U[0]
        h_params["u_y"] = U[1]
        h_params["u_z"] = U[2]
        h_params["v_x"] = V[0]
        h_params["v_y"] = V[1]
        h_params["v_z"] = V[2]
        h_params["w_x"] = W[0]
        h_params["w_y"] = W[1]
        h_params["w_z"] = W[2]
        h_params["sun_dir_x"] = sun_direction[0]
        h_params["sun_dir_y"] = sun_direction[1]
        h_params["sun_dir_z"] = sun_direction[2]
        # Keep the sun's total radiated power constant as sharpness changes.
        # A cosine-power lobe cos^n integrates to 2*pi/(n+1) over the hemisphere,
        # so its integrated power falls as 1/(n+1). Compensate by scaling
        # intensity by (n+1), normalized to the default exponent so `intensity`
        # keeps its meaning at the default Sun Sharpness (15). Only the cos-lobe
        # sky uses sun_exp; leave the directional-sun path untouched.
        eff_sun_intensity = sun_intensity
        if env_sky_nee:
            eff_sun_intensity *= (sun_exp + 1.0) / (15.0 + 1.0)
        h_params["sun_intensity"] = eff_sun_intensity
        h_params["radius_scale"] = radius_scale
        h_params["sky_top_r"] = sky_color_top[0]
        h_params["sky_top_g"] = sky_color_top[1]
        h_params["sky_top_b"] = sky_color_top[2]
        h_params["sky_bot_r"] = sky_color_bottom[0]
        h_params["sky_bot_g"] = sky_color_bottom[1]
        h_params["sky_bot_b"] = sky_color_bottom[2]
        h_params["_pad1"] = 0
        h_params["accum_buffer"] = self._d_accum.data.ptr
        h_params["sample_index"] = self._sample_count
        h_params["samples_accumulated"] = self._sample_count + 1
        h_params["exposure"] = exposure

        # DOF, bounce control, materials
        h_params["aperture"] = aperture
        h_params["focal_plane_depth"] = focal_plane_depth
        if cam_right is not None:
            cr = np.asarray(cam_right, dtype=np.float32)
            h_params["cam_right_x"] = cr[0]
            h_params["cam_right_y"] = cr[1]
            h_params["cam_right_z"] = cr[2]
        if cam_up is not None:
            cu = np.asarray(cam_up, dtype=np.float32)
            h_params["cam_up_x"] = cu[0]
            h_params["cam_up_y"] = cu[1]
            h_params["cam_up_z"] = cu[2]
        h_params["max_bounces"] = max_bounces
        h_params["rr_start_depth"] = rr_start_depth
        h_params["firefly_clamp"] = 1 if firefly_clamp else 0
        h_params["firefly_clamp_max"] = firefly_clamp_max
        h_params["global_material"] = global_material
        h_params["glossy_ior"] = glossy_ior

        # Sun NEE params
        h_params["sun_color_r"] = sun_color[0]
        h_params["sun_color_g"] = sun_color[1]
        h_params["sun_color_b"] = sun_color[2]
        h_params["sun_sampling"] = 1 if sun_sampling else 0
        h_params["_pad4"] = 0

        # Guide buffer pointers (null when not writing guides)
        if write_guides and self._d_albedo is not None:
            h_params["albedo_buffer"] = self._d_albedo.data.ptr
            h_params["normal_buffer"] = self._d_normal.data.ptr
        else:
            h_params["albedo_buffer"] = 0
            h_params["normal_buffer"] = 0

        # Albedo color controls
        h_params["albedo_saturation"] = albedo_saturation
        h_params["albedo_brightness"] = albedo_brightness

        # Sphere size jitter
        h_params["sphere_size_jitter"] = sphere_size_jitter

        # RNG decorrelation: monotonic counter that never resets
        h_params["frame_seed"] = self._global_frame_counter
        self._global_frame_counter += 1

        # SDF scene
        h_params["sdf_enabled"] = 1 if sdf_enabled else 0
        if sdf_enabled and sdf_aabb_min is not None:
            h_params["sdf_aabb_min_x"] = sdf_aabb_min[0]
            h_params["sdf_aabb_min_y"] = sdf_aabb_min[1]
            h_params["sdf_aabb_min_z"] = sdf_aabb_min[2]
            h_params["sdf_aabb_max_x"] = sdf_aabb_max[0]
            h_params["sdf_aabb_max_y"] = sdf_aabb_max[1]
            h_params["sdf_aabb_max_z"] = sdf_aabb_max[2]
            # In curve mode, SDF is the only custom primitive (index 0 in its build input).
            # In sphere mode, SDF is the last primitive (index = entity_count).
            h_params["sdf_prim_index"] = 0 if use_curves else self._entity_count

        # Emissive particles
        h_params["emission_intensity"] = emission_intensity

        # Curve primitives
        h_params["use_curves"] = 1 if use_curves else 0
        h_params["curve_length"] = curve_length
        h_params["curve_r0"] = curve_r0
        h_params["curve_r1"] = curve_r1
        h_params["env_sky_nee"] = 1 if env_sky_nee else 0
        h_params["sun_exp"] = sun_exp

        # Photosphere
        h_params["photosphere_enabled"] = 1 if photosphere else 0
        h_params["_pad_photo"] = 0
        h_params["photosphere_tex"] = self._photo_tex_obj if photosphere else 0
        h_params["photosphere_avg_r"] = photosphere_avg_r
        h_params["photosphere_avg_g"] = photosphere_avg_g
        h_params["photosphere_avg_b"] = photosphere_avg_b

        # Rasterize preset (merged sphere renderer)
        amb = np.asarray(ambient_color, dtype=np.float32)
        h_params["rasterize"] = 1 if rasterize else 0
        h_params["ao_enabled"] = 1 if ao_enabled else 0
        h_params["ao_num_rays"] = ao_num_rays
        h_params["ao_radius"] = ao_radius
        h_params["ao_frame_index"] = ao_frame_index
        h_params["ambient_color_r"] = amb[0]
        h_params["ambient_color_g"] = amb[1]
        h_params["ambient_color_b"] = amb[2]
        # pick_* fields stay zero here (pick_mode=0 => normal render path).

        self._d_params.set(
            np.frombuffer(h_params.tobytes(), dtype=np.uint8)
        )

        # OptiX launch: trace + accumulate into HDR buffer
        optix.launch(
            self._pipeline,
            self._stream,
            self._d_params.data.ptr,
            PARAMS_DTYPE.itemsize,
            self._sbt,
            width,
            height,
            1,  # depth
        )

        self._sample_count += 1

    def _resolve_accum_to_pbo(self, image_ptr, width, height,
                              denoise_enabled, flip_y=True):
        """Run optional denoiser then resolve to linear HDR half-float PBO.

        No tonemapping or gamma is applied — the output is linear HDR in
        rgba16f format, suitable for frame_assembly.frag to tonemap.

        Args:
            image_ptr: Device pointer to the mapped PBO (rgba16f).
            width, height: Image dimensions.
            denoise_enabled: If True, resolve + denoise first, then write
                denoised result to PBO. If False, resolve raw accum directly.
            flip_y: If True, flip Y axis for OpenGL convention (default).
        """
        block = (16, 16, 1)
        grid = ((width + 15) // 16, (height + 15) // 16, 1)
        stream_wrapper = cp.cuda.ExternalStream(self._stream)
        flip_y_val = np.uint32(1 if flip_y else 0)

        if denoise_enabled:
            # Resolve + denoise, then write denoised HDR to PBO
            self._run_denoiser(width, height)
            self._resolve_to_half_kernel(
                grid, block,
                (self._d_denoised.data.ptr,
                 image_ptr,
                 np.uint32(width),
                 np.uint32(height),
                 np.uint32(1),  # already resolved to mean
                 flip_y_val),
                stream=stream_wrapper,
            )
        else:
            # Resolve raw accum buffer directly to half-float PBO
            self._resolve_to_half_kernel(
                grid, block,
                (self._d_accum.data.ptr,
                 image_ptr,
                 np.uint32(width),
                 np.uint32(height),
                 np.uint32(self._sample_count),
                 flip_y_val),
                stream=stream_wrapper,
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, width, height, eye, U, V, W,
               sun_direction=(0.577, 0.577, 0.577),
               sun_intensity=1.0, radius_scale=1.0, exposure=1.0,
               sky_color_top=(0.45, 0.62, 0.85),
               sky_color_bottom=(0.08, 0.08, 0.10),
               aperture=0.0, focal_plane_depth=10.0,
               cam_right=None, cam_up=None,
               max_bounces=0, rr_start_depth=3,
               firefly_clamp=False, firefly_clamp_max=100.0,
               global_material=0, glossy_ior=1.5,
               sun_color=(1.0, 1.0, 1.0), sun_sampling=True,
               denoise_enabled=False,
               albedo_saturation=0.8, albedo_brightness=1.0,
               sphere_size_jitter=0.0, flip_y=True,
               sdf_enabled=False, sdf_aabb_min=None, sdf_aabb_max=None,
               emission_intensity=10.0,
               use_curves=False, curve_length=1.0,
               curve_r0=1.0, curve_r1=0.5,
               env_sky_nee=False,
               sun_exp=15.0,
               photosphere=False,
               photosphere_avg_r=0.0,
               photosphere_avg_g=0.0,
               photosphere_avg_b=0.0,
               rasterize=False,
               ao_enabled=False, ao_num_rays=2,
               ao_radius=0.5, ao_frame_index=0,
               ambient_color=(0.12, 0.12, 0.12)):
        """Render one sample and accumulate into the HDR buffer.

        Each call adds one sample-per-pixel. The displayed result is the
        running average of all accumulated samples, optionally denoised,
        tonemapped and written to the PBO.  Call reset_accumulation() to
        start a fresh render.

        The entity buffer must NOT be mapped by the caller.

        Args:
            width, height: Output image dimensions.
            eye: Camera position (3-tuple or array).
            U: Camera horizontal extent vector (3-tuple or array).
            V: Camera vertical extent vector (3-tuple or array).
            W: Camera center ray direction (3-tuple or array).
            sun_direction: Unit vector toward the sun (3-tuple).
            sun_intensity: Sun radiance multiplier.
            radius_scale: Multiplier on entity size field.
            exposure: Exposure multiplier for tonemapping.
            sky_color_top: Sky gradient top color (3-tuple).
            sky_color_bottom: Sky gradient bottom color (3-tuple).
            aperture: DOF lens radius (0 = pinhole, no DOF).
            focal_plane_depth: DOF focal distance.
            cam_right: Normalized camera right vector (required if aperture > 0).
            cam_up: Normalized camera up vector (required if aperture > 0).
            max_bounces: Max bounce depth (0 = unlimited, RR only).
            rr_start_depth: Depth at which Russian roulette begins.
            firefly_clamp: Enable per-sample firefly clamping.
            firefly_clamp_max: Max per-sample luminance.
            global_material: Material type (0=Lambert, 1=Glossy, 2=Mirror).
            glossy_ior: Index of refraction for glossy Fresnel.
            sun_color: Sun color RGB (3-tuple).
            sun_sampling: Enable NEE shadow rays for direct sun lighting.
            denoise_enabled: Run OptiX AI denoiser after accumulation.

        Returns:
            moderngl.Texture (rgba8) with the tonemapped image.
        """
        if self._gas_handle is None:
            raise RuntimeError(
                "GAS not built. Call build_accel() before render()."
            )

        self._ensure_display(width, height)
        self._ensure_accum_buffers(width, height)
        if denoise_enabled:
            self._setup_denoiser(width, height)

        # Map entity buffer for the OptiX launch (intersection reads it)
        self._ctx.finish()
        entities_ptr, _ = map_resource(self._entity_res)

        try:
            # Map PBO for tonemap kernel to write into
            image_ptr, _ = map_resource(self._pbo_res)

            try:
                check_cuda(cudart.cudaEventRecord(
                    self._evt_render_start, self._stream_obj))

                self._fill_params_and_launch(
                    entities_ptr, width, height, eye, U, V, W,
                    write_guides=denoise_enabled,
                    sun_direction=sun_direction,
                    sun_intensity=sun_intensity,
                    radius_scale=radius_scale,
                    exposure=exposure,
                    sky_color_top=sky_color_top,
                    sky_color_bottom=sky_color_bottom,
                    aperture=aperture,
                    focal_plane_depth=focal_plane_depth,
                    cam_right=cam_right,
                    cam_up=cam_up,
                    max_bounces=max_bounces,
                    rr_start_depth=rr_start_depth,
                    firefly_clamp=firefly_clamp,
                    firefly_clamp_max=firefly_clamp_max,
                    global_material=global_material,
                    glossy_ior=glossy_ior,
                    sun_color=sun_color,
                    sun_sampling=sun_sampling,
                    albedo_saturation=albedo_saturation,
                    albedo_brightness=albedo_brightness,
                    sphere_size_jitter=sphere_size_jitter,
                    sdf_enabled=sdf_enabled,
                    sdf_aabb_min=sdf_aabb_min,
                    sdf_aabb_max=sdf_aabb_max,
                    emission_intensity=emission_intensity,
                    use_curves=use_curves,
                    curve_length=curve_length,
                    curve_r0=curve_r0,
                    curve_r1=curve_r1,
                    env_sky_nee=env_sky_nee,
                    sun_exp=sun_exp,
                    photosphere=photosphere,
                    photosphere_avg_r=photosphere_avg_r,
                    photosphere_avg_g=photosphere_avg_g,
                    photosphere_avg_b=photosphere_avg_b,
                    rasterize=rasterize,
                    ao_enabled=ao_enabled,
                    ao_num_rays=ao_num_rays,
                    ao_radius=ao_radius,
                    ao_frame_index=ao_frame_index,
                    ambient_color=ambient_color,
                )

                self._resolve_accum_to_pbo(
                    image_ptr, width, height,
                    denoise_enabled=denoise_enabled,
                    flip_y=flip_y,
                )

                check_cuda(cudart.cudaEventRecord(
                    self._evt_render_end, self._stream_obj))
                check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))
                self.last_render_ms = check_cuda(
                    cudart.cudaEventElapsedTime(
                        self._evt_render_start, self._evt_render_end
                    )
                )
            finally:
                unmap_resource(self._pbo_res)
        finally:
            unmap_resource(self._entity_res)

        # Blit PBO to texture (fast GPU-to-GPU copy)
        self._tex.write(self._pbo)
        return self._tex

    def pick(self, ray_origin, ray_dir):
        """Trace a single ray into the current GAS and return the first hit.

        Uses the most recently built traversable (the same GAS/IAS the last
        render used). Launches __raygen__rg in pick mode at 1x1: the ray is
        (ray_origin, ray_dir); the raygen writes [prim_index, hit_t] to a small
        device buffer.

        Returns (prim_index, depth) for a particle hit, or None on miss / SDF
        hit / when no GAS is built. prim_index is the entity index.
        """
        if self._traversable_handle is None:
            return None

        ray_origin = np.asarray(ray_origin, dtype=np.float32)
        ray_dir = np.asarray(ray_dir, dtype=np.float32)

        # Lazily allocate the 2-int output buffer (prim_index, hit_t-as-int).
        if getattr(self, "_d_pick_out", None) is None:
            self._d_pick_out = cp.empty(2, dtype=cp.int32)

        # Map entity buffer for the launch (intersection reads it).
        self._ctx.finish()
        entities_ptr, _ = map_resource(self._entity_res)
        try:
            h_params = np.zeros(1, dtype=PARAMS_DTYPE)
            h_params["entities"] = entities_ptr
            h_params["entity_stride"] = self._entity_stride
            h_params["handle"] = self._traversable_handle
            h_params["width"] = 1
            h_params["height"] = 1
            h_params["eye_x"] = ray_origin[0]
            h_params["eye_y"] = ray_origin[1]
            h_params["eye_z"] = ray_origin[2]
            # SDF prim index so the sphere IS can detect the SDF primitive.
            h_params["sdf_enabled"] = 1 if self._sdf_enabled else 0
            h_params["sdf_prim_index"] = self._entity_count
            h_params["radius_scale"] = self._last_radius_scale
            h_params["sphere_size_jitter"] = self._last_sphere_size_jitter
            h_params["pick_mode"] = 1
            h_params["pick_out"] = int(self._d_pick_out.data.ptr)
            h_params["pick_dir_x"] = ray_dir[0]
            h_params["pick_dir_y"] = ray_dir[1]
            h_params["pick_dir_z"] = ray_dir[2]

            self._d_params.set(np.frombuffer(h_params.tobytes(), dtype=np.uint8))

            optix.launch(
                self._pipeline, self._stream, self._d_params.data.ptr,
                PARAMS_DTYPE.itemsize, self._sbt, 1, 1, 1,
            )
            check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))

            out = cp.asnumpy(self._d_pick_out)
        finally:
            unmap_resource(self._entity_res)

        prim = int(out[0])
        if prim < 0:
            return None
        depth = float(np.frombuffer(out[1:2].tobytes(), dtype=np.float32)[0])
        return (prim, depth)

    def render_from_camera(self, width, height, cam_pos, cam_dir, cam_up,
                           fov_deg, sun_direction=(0.577, 0.577, 0.577),
                           sun_intensity=1.0, radius_scale=1.0, exposure=1.0,
                           sky_color_top=(0.45, 0.62, 0.85),
                           sky_color_bottom=(0.08, 0.08, 0.10),
                           aperture=0.0, focal_plane_depth=10.0,
                           max_bounces=0, rr_start_depth=3,
                           firefly_clamp=False, firefly_clamp_max=100.0,
                           global_material=0, glossy_ior=1.5,
                           sun_color=(1.0, 1.0, 1.0), sun_sampling=True,
                           denoise_enabled=False):
        """Convenience: render from FPS camera vectors.

        Converts Fluoddity's ControllerCam-style vectors to OptiX pinhole
        basis and calls render().

        Args:
            cam_pos: Camera position (3,).
            cam_dir: Unit look direction (3,).
            cam_up: Unit up vector (3,).
            fov_deg: Vertical FOV in degrees.
            sun_direction: Unit vector toward the sun (3-tuple).
            sun_intensity: Sun radiance multiplier.
            radius_scale: Multiplier on entity size field.
            exposure: Exposure multiplier for tonemapping.
            sky_color_top: Sky gradient top color (3-tuple).
            sky_color_bottom: Sky gradient bottom color (3-tuple).
            aperture: DOF lens radius (0 = pinhole, no DOF).
            focal_plane_depth: DOF focal distance.
            max_bounces: Max bounce depth (0 = unlimited, RR only).
            rr_start_depth: Depth at which Russian roulette begins.
            firefly_clamp: Enable per-sample firefly clamping.
            firefly_clamp_max: Max per-sample luminance.
            global_material: Material type (0=Lambert, 1=Glossy, 2=Mirror).
            glossy_ior: Index of refraction for glossy Fresnel.
            sun_color: Sun color RGB (3-tuple).
            sun_sampling: Enable NEE shadow rays for direct sun lighting.
            denoise_enabled: Run OptiX AI denoiser after accumulation.

        Returns:
            moderngl.Texture (rgba8).
        """
        aspect = width / max(height, 1)
        eye, U, V, W = _camera_basis_from_vectors(
            cam_pos, cam_dir, cam_up, fov_deg, aspect
        )

        # Compute normalized cam_right and cam_up for DOF lens sampling
        cam_dir_arr = np.asarray(cam_dir, dtype=np.float32)
        cam_up_arr = np.asarray(cam_up, dtype=np.float32)
        cam_right_vec = np.cross(cam_dir_arr, cam_up_arr)
        cam_right_vec = cam_right_vec / max(np.linalg.norm(cam_right_vec), 1e-8)
        cam_up_vec = np.cross(cam_right_vec, cam_dir_arr)
        cam_up_vec = cam_up_vec / max(np.linalg.norm(cam_up_vec), 1e-8)

        return self.render(
            width, height, eye, U, V, W,
            sun_direction=sun_direction,
            sun_intensity=sun_intensity,
            radius_scale=radius_scale,
            exposure=exposure,
            sky_color_top=sky_color_top,
            sky_color_bottom=sky_color_bottom,
            aperture=aperture,
            focal_plane_depth=focal_plane_depth,
            cam_right=cam_right_vec,
            cam_up=cam_up_vec,
            max_bounces=max_bounces,
            rr_start_depth=rr_start_depth,
            firefly_clamp=firefly_clamp,
            firefly_clamp_max=firefly_clamp_max,
            global_material=global_material,
            glossy_ior=glossy_ior,
            sun_color=sun_color,
            sun_sampling=sun_sampling,
            denoise_enabled=denoise_enabled,
        )

    # ------------------------------------------------------------------
    # Realtime / Offline mode API (Step 6)
    # ------------------------------------------------------------------

    def render_realtime(self, width, height, eye, U, V, W,
                        radius_scale=1.0,
                        denoise_enabled=False, reset=True,
                        num_samples=1, flip_y=True,
                        physics_steps=0, accum_slot=0, **render_kwargs):
        """Realtime render with configurable sample count and reset behavior.

        Manages GAS scheduling internally. The entity buffer must reflect
        the current particle positions before calling.

        Args:
            width, height: Output dimensions.
            eye, U, V, W: Camera basis vectors.
            radius_scale: Entity size multiplier.
            denoise_enabled: Run AI denoiser on this frame.
            reset: If True, reset accumulation each frame (default).
                Set False for accumulate mode.
            num_samples: Number of samples to trace this frame (default 1).
            physics_steps: Number of physics steps since last render.
                Entities only move on physics steps, so GAS only needs
                rebuilding when steps > 0.
            **render_kwargs: All other render params (sun, sky, materials,
                DOF, bounce control, etc.).

        Returns:
            moderngl.Texture (rgba8).
        """
        # Select this eye's accumulator (slot 0 = default / non-stereo).
        self.set_accum_slot(accum_slot)

        if reset:
            self.reset_accumulation()

        sphere_size_jitter = render_kwargs.get('sphere_size_jitter', 0.0)
        sdf_enabled = render_kwargs.get('sdf_enabled', False)
        sdf_aabb_min = render_kwargs.get('sdf_aabb_min', None)
        sdf_aabb_max = render_kwargs.get('sdf_aabb_max', None)
        use_curves = render_kwargs.get('use_curves', False)
        curve_length = render_kwargs.get('curve_length', 1.0)
        curve_r0 = render_kwargs.get('curve_r0', 1.0)
        curve_r1 = render_kwargs.get('curve_r1', 0.5)

        # GAS scheduling: always do a full rebuild when entities have moved
        # (physics_steps > 0), the GAS doesn't exist yet, or geometry type
        # changed (curves <-> spheres). Entities only move when physics
        # runs, so skip GAS work entirely when paused.
        if (physics_steps > 0 or self._gas_handle is None
                or use_curves != self._use_curves):
            self.build_accel(radius_scale, sphere_size_jitter,
                             sdf_enabled, sdf_aabb_min, sdf_aabb_max,
                             use_curves, curve_length, curve_r0, curve_r1)

        # Single-sample fast path
        if num_samples == 1:
            return self.render(
                width, height, eye, U, V, W,
                radius_scale=radius_scale,
                denoise_enabled=denoise_enabled,
                flip_y=flip_y,
                **render_kwargs,
            )

        # Multi-sample or denoise-only (num_samples=0) path
        self._ensure_display(width, height)
        self._ensure_accum_buffers(width, height)
        if denoise_enabled:
            self._setup_denoiser(width, height)

        self._ctx.finish()
        entities_ptr, _ = map_resource(self._entity_res)

        try:
            image_ptr, _ = map_resource(self._pbo_res)
            try:
                check_cuda(cudart.cudaEventRecord(
                    self._evt_render_start, self._stream_obj))

                for i in range(num_samples):
                    self._fill_params_and_launch(
                        entities_ptr, width, height, eye, U, V, W,
                        write_guides=(denoise_enabled and i == 0
                                      and self._sample_count == 0),
                        radius_scale=radius_scale,
                        **render_kwargs,
                    )

                self._resolve_accum_to_pbo(
                    image_ptr, width, height,
                    denoise_enabled=denoise_enabled,
                    flip_y=flip_y,
                )

                check_cuda(cudart.cudaEventRecord(
                    self._evt_render_end, self._stream_obj))
                check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))
                self.last_render_ms = check_cuda(
                    cudart.cudaEventElapsedTime(
                        self._evt_render_start, self._evt_render_end
                    )
                )
            finally:
                unmap_resource(self._pbo_res)
        finally:
            unmap_resource(self._entity_res)

        self._tex.write(self._pbo)
        return self._tex

    def render_offline_begin(self, width, height, total_substeps,
                             spp_per_substep, denoise_enabled=False,
                             accum_slot=0):
        """Begin an offline motion-blur render.

        Resets the accumulation buffer and stores parameters for the
        substep sequence. After calling this, call render_offline_substep()
        for each temporal sub-step, then render_offline_finish() to get
        the final tonemapped image.

        Args:
            width, height: Output dimensions.
            total_substeps: Number of temporal sub-steps for motion blur.
                Each sub-step is a GAS refit at a different time instant.
            spp_per_substep: Samples per pixel per sub-step.
            denoise_enabled: Whether to denoise the final composed frame.
            accum_slot: Accumulation slot (stereogram: one per eye; 0 = default).
        """
        self.set_accum_slot(accum_slot)
        self._ensure_display(width, height)
        self._ensure_accum_buffers(width, height)
        self.reset_accumulation()

        self._offline_active = True
        self._offline_width = width
        self._offline_height = height
        self._offline_denoise = denoise_enabled
        self._offline_substeps_done = 0
        self._offline_spp_per_substep = spp_per_substep

        if denoise_enabled:
            self._setup_denoiser(width, height)

    def render_offline_substep(self, eye, U, V, W,
                               radius_scale=1.0, accum_slot=0,
                               spp=None, **render_kwargs):
        """Trace samples for one temporal sub-step.

        By default traces ``spp_per_substep`` samples (set in
        render_offline_begin). Pass ``spp`` to override the count for this
        substep — used by the cadence-lock scheduler, which places a variable
        number of samples on each cadence slot while building the GAS only once.

        The caller must update entity positions (via physics step) and
        ensure the entity buffer reflects the new state BEFORE calling.
        The GAS is rebuilt to match the updated positions. Samples
        accumulate into the same HDR buffer across all sub-steps,
        producing motion blur via temporal integration.

        Does NOT tonemap or denoise — those happen in render_offline_finish().

        Args:
            eye, U, V, W: Camera basis (constant across sub-steps).
            radius_scale: Entity size multiplier.
            **render_kwargs: All other render params (sun, sky, materials,
                DOF, bounce control, etc.).
        """
        if not self._offline_active:
            raise RuntimeError(
                "Call render_offline_begin() before render_offline_substep()."
            )

        # Select this eye's accumulator (stereogram interleaves both eyes).
        self.set_accum_slot(accum_slot)

        w = self._offline_width
        h = self._offline_height

        # GAS update: full rebuild every sub-step (entities moved).
        sphere_size_jitter = render_kwargs.get('sphere_size_jitter', 0.0)
        sdf_enabled = render_kwargs.get('sdf_enabled', False)
        sdf_aabb_min = render_kwargs.get('sdf_aabb_min', None)
        sdf_aabb_max = render_kwargs.get('sdf_aabb_max', None)
        use_curves = render_kwargs.get('use_curves', False)
        curve_length = render_kwargs.get('curve_length', 1.0)
        curve_r0 = render_kwargs.get('curve_r0', 1.0)
        curve_r1 = render_kwargs.get('curve_r1', 0.5)
        self.build_accel(radius_scale, sphere_size_jitter,
                         sdf_enabled, sdf_aabb_min, sdf_aabb_max,
                         use_curves, curve_length, curve_r0, curve_r1)

        # Map entity buffer for all SPP in this substep
        self._ctx.finish()
        entities_ptr, _ = map_resource(self._entity_res)

        try:
            # Only write guide buffers on the first sample of the first
            # substep. Later samples/substeps pass null pointers so the
            # raygen skips guide writes, keeping a single coherent snapshot
            # for the denoiser.
            first_substep = (self._offline_substeps_done == 0)
            n_spp = self._offline_spp_per_substep if spp is None else max(1, spp)

            check_cuda(cudart.cudaEventRecord(
                self._evt_render_start, self._stream_obj))

            for i in range(n_spp):
                self._fill_params_and_launch(
                    entities_ptr, w, h, eye, U, V, W,
                    write_guides=(self._offline_denoise
                                  and first_substep and i == 0),
                    radius_scale=radius_scale,
                    **render_kwargs,
                )

            check_cuda(cudart.cudaEventRecord(
                self._evt_render_end, self._stream_obj))
            check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))
            self.last_render_ms = check_cuda(
                cudart.cudaEventElapsedTime(
                    self._evt_render_start, self._evt_render_end
                )
            )
        finally:
            unmap_resource(self._entity_res)

        self._offline_substeps_done += 1

    def render_offline_finish(self, flip_y=True, accum_slot=0,
                              end_session=True):
        """Denoise (if enabled) and resolve to linear HDR texture.

        Must be called after all sub-steps are complete. Returns the
        resolved HDR texture (rgba16f) for frame_assembly.frag to tonemap.

        Args:
            flip_y: If True, flip Y axis for OpenGL convention (default).
            accum_slot: Accumulation slot to resolve (stereogram: per eye).
            end_session: If True, mark the offline render complete. Set False
                for the first eye of a stereogram pair so the second eye can
                still resolve its slot before the session ends.

        Returns:
            moderngl.Texture (rgba16f) with linear HDR data.
        """
        if not self._offline_active:
            raise RuntimeError(
                "No offline render in progress."
            )

        self.set_accum_slot(accum_slot)

        w = self._offline_width
        h = self._offline_height

        self._ensure_display(w, h)

        self._ctx.finish()
        image_ptr, _ = map_resource(self._pbo_res)

        try:
            self._resolve_accum_to_pbo(
                image_ptr, w, h,
                denoise_enabled=self._offline_denoise,
                flip_y=flip_y,
            )
            check_cuda(cudart.cudaStreamSynchronize(self._stream_obj))
        finally:
            unmap_resource(self._pbo_res)

        self._tex.write(self._pbo)
        if end_session:
            self._offline_active = False
        return self._tex

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def update_entity_buffer(self, entity_buffer, entity_count):
        """Re-register a new entity buffer (after world_size change / reset).

        Args:
            entity_buffer: New ModernGL buffer.
            entity_count: New entity count.
        """
        # Unregister old buffer
        if self._entity_res is not None:
            unregister_resource(self._entity_res)

        self._entity_buffer = entity_buffer
        self._entity_count = entity_count

        flags = cudart.cudaGraphicsRegisterFlags
        self._entity_res = register_gl_buffer(
            entity_buffer, flags.cudaGraphicsRegisterFlagsReadOnly
        )

        # Invalidate GAS (must be rebuilt)
        self._gas_handle = None
        self._d_gas = None
        self._d_temp = None
        self._d_aabbs = None

        # Reset accumulation (scene changed)
        self.reset_accumulation()

    def resize(self, width, height):
        """Force recreation of PBO and output texture for a new resolution."""
        self._render_width = 0  # force _ensure_display to recreate
        self._render_height = 0

    def get_guide_buffers(self):
        """Return (albedo, normal) as numpy float32 arrays, shape (H, W, 4).

        For diagnostic visualization of the denoiser guide buffers.
        Returns (None, None) if buffers are not allocated.
        """
        if self._d_albedo is None or self._d_normal is None:
            return None, None
        w, h = self._accum_width, self._accum_height
        albedo = cp.asnumpy(self._d_albedo).reshape(h, w, 4)
        normal = cp.asnumpy(self._d_normal).reshape(h, w, 4)
        return albedo, normal

    # ------------------------------------------------------------------
    # Photosphere texture
    # ------------------------------------------------------------------

    def set_photosphere_texture(self, image_data):
        """Create a CUDA texture object from equirectangular image data.

        Args:
            image_data: dict with 'data' (H,W,4 float16 RGBA),
                        'width', 'height'.
                        Pass None to destroy existing texture.
        """
        self._destroy_photosphere_texture()

        if image_data is None:
            return

        width = image_data['width']
        height = image_data['height']
        rgba_f16 = image_data['data']  # (H, W, 4) float16

        # Channel format: 4x 16-bit float
        chan_desc = check_cuda(cudart.cudaCreateChannelDesc(
            16, 16, 16, 16,
            cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat))

        # Allocate CUDA array
        self._photo_cuda_array = check_cuda(
            cudart.cudaMallocArray(chan_desc, width, height, 0))

        # Upload data: cudaMemcpy2DToArray(dst, wOff, hOff, src, spitch, w_bytes, h, kind)
        src_pitch = width * 4 * 2  # 4 channels * 2 bytes (float16)
        check_cuda(cudart.cudaMemcpy2DToArray(
            self._photo_cuda_array,
            0, 0,
            rgba_f16.ctypes.data,
            src_pitch,
            src_pitch,
            height,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

        # Create texture object
        res_desc = cudart.cudaResourceDesc()
        res_desc.resType = cudart.cudaResourceType.cudaResourceTypeArray
        res_desc.res.array.array = self._photo_cuda_array

        tex_desc = cudart.cudaTextureDesc()
        tex_desc.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeWrap
        tex_desc.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeClamp
        tex_desc.filterMode = cudart.cudaTextureFilterMode.cudaFilterModeLinear
        tex_desc.readMode = cudart.cudaTextureReadMode.cudaReadModeElementType
        tex_desc.normalizedCoords = 1

        self._photo_tex_obj = check_cuda(
            cudart.cudaCreateTextureObject(res_desc, tex_desc, None))
        self._photo_loaded = True

    def _destroy_photosphere_texture(self):
        """Destroy CUDA texture object and array."""
        if self._photo_tex_obj != 0:
            try:
                cudart.cudaDestroyTextureObject(self._photo_tex_obj)
            except Exception:
                pass
            self._photo_tex_obj = 0
        if self._photo_cuda_array is not None:
            try:
                cudart.cudaFreeArray(self._photo_cuda_array)
            except Exception:
                pass
            self._photo_cuda_array = None
        self._photo_loaded = False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release all CUDA and OptiX resources."""
        self._destroy_photosphere_texture()

        if self._pbo_res is not None:
            try:
                unregister_resource(self._pbo_res)
            except RuntimeError:
                pass
            self._pbo_res = None

        if self._entity_res is not None:
            try:
                unregister_resource(self._entity_res)
            except RuntimeError:
                pass
            self._entity_res = None

        # Destroy CUDA stream and events
        if hasattr(self, '_stream_obj') and self._stream_obj is not None:
            try:
                cudart.cudaStreamDestroy(self._stream_obj)
            except RuntimeError:
                pass
            self._stream_obj = None
            self._stream = None
        for attr in ('_evt_gas_start', '_evt_gas_end',
                     '_evt_render_start', '_evt_render_end'):
            evt = getattr(self, attr, None)
            if evt is not None:
                try:
                    cudart.cudaEventDestroy(evt)
                except RuntimeError:
                    pass
                setattr(self, attr, None)

        self._gas_handle = None
        self._d_gas = None
        self._d_temp = None
        self._d_aabbs = None
        self._d_params = None
        self._d_accum = None
        self._d_albedo = None
        self._d_normal = None
        self._sbt_mem = None

        # Denoiser resources
        self._denoiser = None
        self._d_denoiser_state = None
        self._d_denoiser_scratch = None
        self._d_resolved = None
        self._d_denoised = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    @staticmethod
    def is_available():
        """Check if OptiX/CUDA/RTX hardware is available.

        Returns False gracefully on machines without the required
        packages or hardware.
        """
        try:
            import optix as _optix  # noqa: F811
        except ImportError:
            print("PathTracer unavailable: 'optix' package not installed")
            return False
        try:
            import cupy as _cp  # noqa: F811
        except ImportError:
            print("PathTracer unavailable: 'cupy' package not installed")
            return False
        try:
            from cuda.bindings import runtime as _cudart  # noqa: F811
        except ImportError:
            print("PathTracer unavailable: 'cuda-python' package not installed")
            return False
        try:
            err, count = _cudart.cudaGetDeviceCount()
            if err != _cudart.cudaError_t.cudaSuccess or count == 0:
                print("PathTracer unavailable: no CUDA-capable GPU detected")
                return False
            return True
        except Exception as e:
            print(f"PathTracer unavailable: {e}")
            return False
