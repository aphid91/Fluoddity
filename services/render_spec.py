"""
RenderSpec: complete snapshot of app state for scheduled video rendering.

A RenderSpec captures everything needed to reproduce an exact simulation state:
physics config, camera, editor state (preferences + imgui layout), and GPU buffer
snapshots (entities, canvas). It is a thin container that bundles the three
independent save systems:

    - Physics  -> services/config_saver.py  (PhysicsConfig)
    - Editor   -> services/editor_saver.py  (EditorSave: prefs + imgui layout)
    - Sim state-> services/simulation_saver.py (entity + canvas GPU buffers)

plus the small .frs-specific camera / controller-cam / sim-metadata dicts.

File format:
    RenderSpecs/
        MyRender.frs/               # .frs = Fluoddity Render Spec (directory)
            metadata.json           # All scalar/dict state (incl. editor block)
            entities.npz            # Compressed entity buffer
            canvas.npz              # Compressed 3D canvas texture

Versioning:
    v1 (legacy): preferences stored as a FLAT dict, no imgui layout.
    v2 (current): editor block { preferences (nested), imgui_layout }.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from services.config_saver import ConfigSaver, PhysicsConfig
from services.editor_saver import EditorSaver, EditorSave
from services.simulation_saver import SimulationSaver
from utilities.paths import get_render_specs_dir

RENDER_SPEC_VERSION = 2


@dataclass
class RenderSpec:
    """Complete specification for a scheduled video render."""
    display_name: str = ""
    physics_config_dict: dict = field(default_factory=dict)
    camera_state: dict = field(default_factory=dict)
    controller_cam_state: dict = field(default_factory=dict)
    editor_save: EditorSave | None = None
    sim_metadata: dict = field(default_factory=dict)
    version: int = RENDER_SPEC_VERSION
    # Path to the .frs directory on disk (set after save/load)
    dir_path: Path | None = None


class RenderSpecService:
    """Service for capturing, saving, loading, and applying RenderSpec snapshots."""

    def __init__(self, editor_saver: EditorSaver | None = None,
                 simulation_saver: SimulationSaver | None = None):
        self.editor_saver = editor_saver or EditorSaver()
        self.simulation_saver = simulation_saver or SimulationSaver()

    def capture_current_state(self, sim, camera, controller_cam, ui_state,
                              config_saver: ConfigSaver, rule_manager,
                              name: str) -> tuple[RenderSpec, dict]:
        """Snapshot all app state + GPU buffers into a RenderSpec.

        Returns (spec, gpu_buffers) where gpu_buffers is a dict of numpy arrays.
        """
        # 1. Physics config (reuse existing serialization)
        rule = rule_manager.get_current_rule()
        physics_config = config_saver.create_config(ui_state.sim, rule, None)
        physics_config_dict = physics_config.to_dict()

        # 2. Camera state (.frs-specific — small, kept inline)
        cam = ui_state.camera
        camera_state = {
            'position': cam.position.tolist(),
            'zoom': cam.zoom,
            'orbit_center': cam.orbit_center.tolist(),
            'orbit_rate': cam.orbit_rate,
            'orbit_angle': cam.orbit_angle,
            'orbit_pitch': cam.orbit_pitch,
            'fov': cam.fov,
            'aperture': cam.aperture,
            'focal_plane_depth': cam.focal_plane_depth,
            'move_speed': cam.move_speed,
            'rotate_speed': cam.rotate_speed,
            'stereogram': cam.stereogram,
            'eye_offset': cam.eye_offset,
            'stereo_toe_in': cam.stereo_toe_in,
            'stereo_wall_eye': cam.stereo_wall_eye,
        }

        # 3. Controller cam state (FPS camera)
        controller_cam_state = {
            'pos': controller_cam.pos.tolist(),
            'yaw': controller_cam.yaw,
            'pitch': controller_cam.pitch,
            'fov': controller_cam.fov,
        }

        # 4. Editor state (preferences + imgui layout)
        editor_save = self.editor_saver.create_save(ui_state.preferences)

        # 5. Sim metadata
        sim_metadata = self.simulation_saver.sim_metadata(sim)

        # 6. GPU buffer snapshots (entities + canvas). Reading + compressing these
        # is expensive, so skip them entirely when the sim hasn't advanced
        # (frame_count == 0). A frame-0 spec re-seeds particles on its first step
        # anyway, so the dump would be redundant. `has_sim_state` tells the load
        # path whether buffers are present.
        has_sim_state = sim.frame_count != 0
        sim_metadata['has_sim_state'] = has_sim_state
        gpu_buffers = self.simulation_saver.read_buffers(sim) if has_sim_state else {}

        spec = RenderSpec(
            display_name=name,
            physics_config_dict=physics_config_dict,
            camera_state=camera_state,
            controller_cam_state=controller_cam_state,
            editor_save=editor_save,
            sim_metadata=sim_metadata,
            version=RENDER_SPEC_VERSION,
        )

        return spec, gpu_buffers

    def path_for(self, name: str) -> Path:
        """Resolve the .frs directory path a spec with this name would use."""
        return get_render_specs_dir() / f"{name}.frs"

    def spec_exists(self, name: str) -> bool:
        """True if a render spec with this name already exists on disk."""
        return self.path_for(name).exists()

    def save_to_disk(self, spec: RenderSpec, gpu_buffers: dict,
                     dir_path: Path | None = None) -> Path:
        """Save a RenderSpec to a .frs directory on disk. Returns the directory path."""
        if dir_path is None:
            dir_path = get_render_specs_dir() / f"{spec.display_name}.frs"

        dir_path.mkdir(parents=True, exist_ok=True)

        editor_block = (self.editor_saver.to_dict(spec.editor_save)
                        if spec.editor_save is not None else {})
        metadata = {
            'version': RENDER_SPEC_VERSION,
            'display_name': spec.display_name,
            'physics_config': spec.physics_config_dict,
            'camera_state': spec.camera_state,
            'controller_cam_state': spec.controller_cam_state,
            'editor': editor_block,
            'sim_metadata': spec.sim_metadata,
        }
        (dir_path / 'metadata.json').write_text(json.dumps(metadata, indent=2))

        # GPU buffers -> entities.npz + canvas.npz (delegated to SimulationSaver).
        # Omitted for frame-0 specs (no sim state captured — see capture_current_state).
        if gpu_buffers:
            self.simulation_saver.save_buffers(gpu_buffers, dir_path)
            raw_entities = gpu_buffers['entities'].nbytes
            raw_canvas = gpu_buffers['can_packed'].nbytes
            comp_entities = (dir_path / 'entities.npz').stat().st_size
            comp_canvas = (dir_path / 'canvas.npz').stat().st_size
            print(f"[RenderSpec] Saved: {spec.display_name}")
            print(f"  Entities: {raw_entities / 1e6:.1f} MB -> {comp_entities / 1e6:.1f} MB")
            print(f"  Canvas:   {raw_canvas / 1e6:.1f} MB -> {comp_canvas / 1e6:.1f} MB")
        else:
            print(f"[RenderSpec] Saved: {spec.display_name} (no sim state — frame 0)")

        spec.dir_path = dir_path
        return dir_path

    def load_metadata(self, dir_path: Path) -> RenderSpec | None:
        """Load only metadata from a .frs directory (fast, no GPU buffers)."""
        metadata_path = dir_path / 'metadata.json'
        if not metadata_path.exists():
            print(f"[RenderSpec] No metadata.json found in {dir_path}")
            return None

        try:
            data = json.loads(metadata_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[RenderSpec] Failed to load metadata from {dir_path}: {e}")
            return None

        version = data.get('version', 1)
        editor_save = self._editor_save_from_metadata(data, version)

        return RenderSpec(
            display_name=data.get('display_name', dir_path.stem),
            physics_config_dict=data.get('physics_config', {}),
            camera_state=data.get('camera_state', {}),
            controller_cam_state=data.get('controller_cam_state', {}),
            editor_save=editor_save,
            sim_metadata=data.get('sim_metadata', {}),
            version=version,
            dir_path=dir_path,
        )

    def _editor_save_from_metadata(self, data: dict, version: int) -> EditorSave:
        """Build an EditorSave from .frs metadata, handling v1 vs v2 layout.

        v2 stores an `editor` block { preferences (nested), imgui_layout }.
        v1 stored a top-level `preferences` as a FLAT dict and no imgui layout;
        preferences_from_dict() detects flat vs nested, so passing the flat dict
        straight through migrates it correctly.
        """
        if 'editor' in data:
            return self.editor_saver.from_dict(data['editor'])
        # Legacy v1: top-level flat preferences, no layout.
        return EditorSave(
            version=1,
            preferences=data.get('preferences', {}),
            imgui_layout="",
        )

    def load_gpu_buffers(self, dir_path: Path) -> dict | None:
        """Load compressed GPU buffer data from a .frs directory.

        Returns an empty dict for specs saved with no sim state (frame-0 specs,
        where the .npz buffers were intentionally omitted); None only on genuine
        load failure so callers can distinguish "nothing to restore" from "error".
        """
        # A frame-0 spec has no entities.npz/canvas.npz. Detect that up front via
        # the metadata flag (fall back to file presence for older specs) so a
        # missing buffer isn't treated as a failure.
        spec = self.load_metadata(dir_path)
        has_sim_state = True
        if spec is not None and spec.sim_metadata:
            has_sim_state = spec.sim_metadata.get('has_sim_state', True)
        if not has_sim_state or not (dir_path / 'entities.npz').exists():
            return {}
        return self.simulation_saver.load_buffers(dir_path)

    def apply_state(self, spec: RenderSpec, gpu_buffers: dict,
                    sim, camera, controller_cam, ui_state,
                    config_saver: ConfigSaver, rule_manager,
                    apply_editor_visibility: bool = True) -> bool:
        """Apply a RenderSpec's state + GPU buffers to the running app.

        Restores physics, camera, editor state (preferences + imgui layout), and
        the GPU buffers.

        Args:
            apply_editor_visibility: if False, keep current window visibility
                (used by batch/headless render paths). imgui layout is likewise
                skipped in that case to avoid disturbing a headless layout.

        Returns:
            True if entity_count/canvas_resolution changed (buffers reallocated).
        """
        # 1. Apply physics config (includes rule)
        physics_config = PhysicsConfig.from_dict(spec.physics_config_dict)
        rule = config_saver.apply_config(physics_config, ui_state.sim)
        rule_manager.push_rule(rule, ui_state.sim.rule_seed)
        sim.apply_rule(rule)

        # 2. Apply camera state
        cam_data = spec.camera_state
        if cam_data:
            ui_state.camera.position[:] = cam_data.get('position', [0.0, 0.0])
            ui_state.camera.zoom = cam_data.get('zoom', 1.0)
            ui_state.camera.orbit_center[:] = cam_data.get('orbit_center', [0.0, 0.0, 0.0])
            ui_state.camera.orbit_rate = cam_data.get('orbit_rate', 0.0)
            ui_state.camera.orbit_angle = cam_data.get('orbit_angle', 0.0)
            ui_state.camera.orbit_pitch = cam_data.get('orbit_pitch', 0.0)
            ui_state.camera.fov = cam_data.get('fov', 50.0)
            ui_state.camera.aperture = cam_data.get('aperture', 0.0)
            ui_state.camera.focal_plane_depth = cam_data.get('focal_plane_depth', 5.0)
            ui_state.camera.move_speed = cam_data.get('move_speed', 2.0)
            ui_state.camera.rotate_speed = cam_data.get('rotate_speed', 2.0)
            ui_state.camera.stereogram = cam_data.get('stereogram', False)
            ui_state.camera.eye_offset = cam_data.get('eye_offset', 0.1)
            ui_state.camera.stereo_toe_in = cam_data.get('stereo_toe_in', False)
            ui_state.camera.stereo_wall_eye = cam_data.get('stereo_wall_eye', False)

        # 3. Apply controller cam state (FPS camera)
        ccam_data = spec.controller_cam_state
        if ccam_data and controller_cam is not None:
            controller_cam.pos[:] = ccam_data.get('pos', [0.0, 0.0, -3.0])
            controller_cam.yaw = ccam_data.get('yaw', 0.0)
            controller_cam.pitch = ccam_data.get('pitch', 0.0)
            controller_cam.fov = ccam_data.get('fov', 50.0)
            controller_cam._update_vectors()

        # 4. Apply editor state (preferences + imgui layout)
        if spec.editor_save is not None:
            self.editor_saver.apply_save(
                spec.editor_save, ui_state.preferences,
                apply_visibility=apply_editor_visibility,
                apply_layout=apply_editor_visibility,
            )

        # 5. Restore GPU buffers (+ world-size realloc + sim metadata)
        # canvas_resolution lives in prefs; hand it to the sim saver via metadata.
        sim_metadata = dict(spec.sim_metadata) if spec.sim_metadata else {}
        sim_metadata['canvas_resolution'] = ui_state.preferences.rendering.canvas_resolution
        if gpu_buffers:
            world_size_changed = self.simulation_saver.write_buffers(
                sim, gpu_buffers, sim_metadata, rule=rule)
        else:
            # Frame-0 spec: no sim buffers were saved. Skip the buffer restore
            # entirely; force frame_count to 0 so the sim re-seeds particles on
            # its first step. Editor prefs (applied above) already carry any
            # entity_count/canvas_resolution change, but with no buffers to write
            # we still need the world size to match — reallocate if it differs.
            world_size_changed = False
            spec_entity_count = sim_metadata.get('entity_count', sim.entity_count)
            spec_canvas_res = sim_metadata.get('canvas_resolution', sim.canvas_resolution)
            if (spec_entity_count != sim.entity_count
                    or spec_canvas_res != sim.canvas_resolution):
                sim._entity_count = spec_entity_count
                sim.canvas_resolution = spec_canvas_res
                sim.setup_simulation_state()
                sim.setup_shaders()
                sim.apply_rule(rule)
                world_size_changed = True
            sim.frame_count = 0

        return world_size_changed

    def list_available_specs(self) -> list[Path]:
        """List all .frs directories in the RenderSpecs folder."""
        specs_dir = get_render_specs_dir()
        if not specs_dir.exists():
            return []
        dirs = [d for d in specs_dir.iterdir()
                if d.is_dir() and d.suffix == '.frs']
        dirs.sort(key=lambda p: p.name.lower())
        return dirs
